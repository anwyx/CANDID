"""Independent MLP with separate disturbance regression and event logits."""
from __future__ import annotations

import torch
from torch import nn


class DisturbancePredictor(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, target_dim: int = 4,
        obs_horizon: int = 2, action_horizon: int = 8, hidden: int = 256,
        label_horizon: int | None = None,
    ):
        super().__init__()
        if target_dim != 4:
            raise ValueError("Predictor uses two DR values and two event probabilities")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.target_dim = target_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.label_horizon = action_horizon if label_horizon is None else label_horizon
        if not (1 <= self.label_horizon <= action_horizon):
            raise ValueError("label_horizon must be within the action horizon")
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
        # For prefix DR labels, the last actions have no causal effect on
        # the supervised outcome. Mask in NORMALIZED space (zero = mean)
        # so the model cannot learn a spurious dependency on future replans.
        if self.label_horizon < self.action_horizon:
            actions = torch.cat((
                actions[:, :self.label_horizon],
                torch.zeros_like(actions[:, self.label_horizon:]),
            ), dim=1)
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
