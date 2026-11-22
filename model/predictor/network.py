"""Independent MLP with separate disturbance regression and event logits."""
from __future__ import annotations

import torch
from torch import nn


class DisturbancePredictor(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, target_dim: int = 4,
        obs_horizon: int = 2, action_horizon: int = 8, hidden: int = 256,
    ):
        super().__init__()
        if target_dim != 4:
            raise ValueError("Predictor uses two DR values and two event probabilities")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.target_dim = target_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        width = obs_horizon * obs_dim + action_horizon * action_dim
        self.encoder = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.dr_head = nn.Linear(hidden, 2)
        self.event_head = nn.Linear(hidden, 2)

    def forward(self, obs, actions):
        batch = obs.shape[0]
        if obs.shape != (batch, self.obs_horizon, self.obs_dim):
            raise ValueError(f"Expected obs [B,{self.obs_horizon},{self.obs_dim}], got {tuple(obs.shape)}")
        if actions.shape != (batch, self.action_horizon, self.action_dim):
            raise ValueError(
                f"Expected actions [B,{self.action_horizon},{self.action_dim}], got {tuple(actions.shape)}"
            )
        features = torch.cat((obs.reshape(batch, -1), actions.reshape(batch, -1)), dim=-1)
        shared = self.encoder(features)
        return {"dr": self.dr_head(shared), "event_logits": self.event_head(shared)}

    def loss(self, obs, actions, targets, event_mask=None, event_pos_weight=None):
        """Targets are normalized DR (first two) and binary events (last two)."""
        predictions = self.forward(obs, actions)
        if targets.shape != (obs.shape[0], 4):
            raise ValueError("Expected [B,4] targets")
        dr_loss = nn.functional.mse_loss(predictions["dr"], targets[:, :2])
        if event_mask is None:
            event_mask = torch.ones_like(targets[:, 2:])
        if event_pos_weight is None:
            event_pos_weight = torch.ones(2, device=targets.device)
        event_losses = nn.functional.binary_cross_entropy_with_logits(
            predictions["event_logits"], targets[:, 2:],
            pos_weight=event_pos_weight, reduction="none",
        )
        known = event_mask.sum()
        event_loss = (event_losses * event_mask).sum() / known.clamp_min(1)
        return dr_loss + event_loss


class OrdinalDisturbancePredictor(DisturbancePredictor):
    """Predict monotone exceedance probabilities for local DR Mean/Max.

    The scalar score and ordered cutpoints are optimized jointly. The
    differentiable pseudo-DR integrates exceedance probabilities over a
    discrete threshold grid and is capped near its largest threshold;
    it is NOT an exact continuous DR estimate.
    """
    dr_objective = "ordinal"

    def __init__(self, *args, dr_thresholds=(0.0001, 0.001, 0.005, 0.02, 0.05, 0.1),
                 ordinal_pos_weight=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        import math
        values = tuple(float(t) for t in dr_thresholds)
        if not values or any(not math.isfinite(t) or t <= 0 for t in values) or any(
            a >= b for a, b in zip(values, values[1:])
        ):
            raise ValueError("Thresholds must be positive and strictly increasing")
        self.register_buffer("dr_thresholds", torch.tensor(values, dtype=torch.float32))
        self.ordinal_pos_weight = float(ordinal_pos_weight)
        if not math.isfinite(self.ordinal_pos_weight) or self.ordinal_pos_weight <= 0:
            raise ValueError("ordinal_pos_weight must be positive")
        self.dr_head = nn.Linear(self.encoder[2].out_features, 2)
        self.first_cutpoint = nn.Parameter(torch.full((2, 1), 1.5))
        self.raw_cutpoint_gaps = nn.Parameter(
            torch.full((2, len(values)-1), -0.3)
        )

    def forward(self, obs, actions):
        batch = obs.shape[0]
        if obs.shape != (batch, self.obs_horizon, self.obs_dim):
            raise ValueError("Invalid observation shape")
        if actions.shape != (batch, self.action_horizon, self.action_dim):
            raise ValueError("Invalid action shape")
        features = torch.cat((obs.reshape(batch, -1), actions.reshape(batch, -1)), dim=-1)
        shared = self.encoder(features)
        score = self.dr_head(shared).unsqueeze(-1)
        cutpoints = torch.cat((
            self.first_cutpoint,
            self.first_cutpoint + nn.functional.softplus(self.raw_cutpoint_gaps).cumsum(dim=-1),
        ), dim=-1)
        logits = score - cutpoints.unsqueeze(0)
        probabilities = torch.sigmoid(logits)
        increments = torch.diff(
            torch.cat((self.dr_thresholds.new_zeros((1,)), self.dr_thresholds))
        )
        pseudo_dr = (probabilities * increments).sum(dim=-1)
        return {
            "dr": pseudo_dr,
            "ordinal_logits": logits,
            "event_logits": self.event_head(shared),
        }

    def loss(self, obs, actions, targets, event_mask=None, event_pos_weight=None):
        """Targets contain PHYSICAL-UNIT DR and unnormalized event labels."""
        predictions = self.forward(obs, actions)
        if targets.shape != (obs.shape[0], 4):
            raise ValueError("Expected [B,4] physical targets")
        labels = (targets[:, :2, None] > self.dr_thresholds).to(targets.dtype)
        positive_weight = torch.as_tensor(
            self.ordinal_pos_weight, dtype=targets.dtype, device=targets.device
        )
        dr_loss = nn.functional.binary_cross_entropy_with_logits(
            predictions["ordinal_logits"], labels, pos_weight=positive_weight
        )
        if event_mask is None:
            event_mask = torch.ones_like(targets[:, 2:])
        if event_pos_weight is None:
            event_pos_weight = torch.ones(2, device=targets.device)
        event_losses = nn.functional.binary_cross_entropy_with_logits(
            predictions["event_logits"], targets[:, 2:],
            pos_weight=event_pos_weight, reduction="none",
        )
        event_loss = (event_losses * event_mask).sum() / event_mask.sum().clamp_min(1)
        return dr_loss + event_loss
