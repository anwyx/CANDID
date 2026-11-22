"""Action-conditioned multi-horizon consequence predictor for scene1.

A state-only baseline (which may use velocities) plus an action-dependent
branch (whose object velocities are masked).  Guidance uses the *same*
multi-horizon scores that counterfactual ranking trains.

Inputs are NORMALIZED features. Action features are 6 delta-q commands and one
absolute gripper command, normalized with v3 action statistics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.predictor.action_features import action_observation_mask

CONSEQUENCE_HORIZONS = (4, 8, 16, 32)
ACTION_HORIZON = 8
OBS_HORIZON = 2


class MultiHorizonConsequencePredictor(nn.Module):
    dr_objective = "multi_ordinal"

    def __init__(self, obs_dim=151, action_dim=7, hidden=256,
                 dr_thresholds=(.0001,.0005,.001,.005,.01,.05,.1,.2,.5,1.),
                 ordinal_pos_weight=5.0, horizon_weights=(.25,.25,.25,.25)):
        super().__init__()
        if obs_dim != 151 or action_dim != 7:
            raise ValueError("Scene1 v3 requires obs_dim=151, action_dim=7")
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.obs_horizon, self.action_horizon = OBS_HORIZON, ACTION_HORIZON
        self.hidden = int(hidden)
        self.ordinal_pos_weight = float(ordinal_pos_weight)
        thresholds = torch.as_tensor(dr_thresholds, dtype=torch.float32)
        if (thresholds.ndim != 1 or thresholds.numel() == 0
                or torch.any(thresholds <= 0)
                or torch.any(thresholds[1:] <= thresholds[:-1])):
            raise ValueError("DR thresholds must be strictly increasing positive values")
        weights = torch.as_tensor(horizon_weights, dtype=torch.float32)
        if (weights.shape != (4,) or not torch.isfinite(weights).all()
                or (weights < 0).any() or float(weights.sum()) <= 0):
            raise ValueError("Require four nonnegative horizon weights with positive sum")
        self.register_buffer("dr_thresholds", thresholds)
        self.register_buffer("horizon_weights", weights / weights.sum())
        self.register_buffer("action_obs_mask",
                             torch.as_tensor(action_observation_mask()))
        self.state_encoder = nn.Sequential(
            nn.Linear(2 * obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(2 * obs_dim + 8 * action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        # The state-only branch cannot create an action gradient. All action
        # derivatives flow through the action branch.
        self.state_dr = nn.Linear(hidden, 8)
        self.state_event = nn.Linear(hidden, 8)
        # action_encoder emits one feature vector PER horizon [B, 4, hidden].
        # Predict only DR_mean/DR_max and Topple/Fall for that horizon.
        self.action_dr = nn.Linear(hidden, 2)
        self.action_event = nn.Linear(hidden, 2)
        self.first_cutpoint = nn.Parameter(torch.full((4, 2, 1), 1.5))
        self.raw_cutpoint_gaps = nn.Parameter(
            torch.full((4, 2, max(len(dr_thresholds)-1, 0)), -.3))

    def forward(self, obs, actions):
        n = obs.shape[0]
        if obs.shape != (n, 2, 151) or actions.shape != (n, 8, 7):
            raise ValueError("Expected normalized obs [B,2,151], action [B,8,7]")
        state = self.state_encoder(obs.reshape(n, -1))
        masked_obs = obs * self.action_obs_mask.view(1, 1, -1)

        # H=4 must not see the four commands that will not be executed.
        action_inputs = actions.unsqueeze(1).expand(-1, 4, -1, -1).clone()
        action_inputs[:, 0, 4:, :] = 0.
        expanded_obs = masked_obs.reshape(n, 1, -1).expand(-1, 4, -1)
        joint = torch.cat((expanded_obs, action_inputs.reshape(n, 4, -1)), dim=-1)
        act_features = self.action_encoder(joint.reshape(n*4, -1)).reshape(n, 4, -1)

        score = (self.state_dr(state).reshape(n, 4, 2)
                 + self.action_dr(act_features).reshape(n, 4, 2))
        event_logits = (self.state_event(state).reshape(n, 4, 2)
                        + self.action_event(act_features).reshape(n, 4, 2))
        cutpoints = torch.cat((
            self.first_cutpoint,
            self.first_cutpoint + F.softplus(self.raw_cutpoint_gaps).cumsum(dim=-1)),
            dim=-1)
        ordinal_logits = score.unsqueeze(-1) - cutpoints.unsqueeze(0)
        increments = torch.diff(torch.cat((
            self.dr_thresholds.new_zeros(1), self.dr_thresholds)))
        multi_dr = (torch.sigmoid(ordinal_logits) * increments).sum(dim=-1)

        # Legacy inference can inspect expected combined risk indicators.
        # Guidance instead uses each separate physical horizon below.
        weights = self.horizon_weights.view(1, 4, 1)
        dr = (multi_dr * weights).sum(dim=1)
        event_prob = (torch.sigmoid(event_logits) * weights).sum(dim=1)
        event_logits_avg = torch.logit(event_prob.clamp(1e-6, 1.-1e-6))
        return {"dr": dr, "event_logits": event_logits_avg,
                "multi_dr": multi_dr,
                "multi_event_logits": event_logits,
                "multi_ordinal_logits": ordinal_logits,
                "risk_horizon_weights": self.horizon_weights,
                "ordinal_logits": ordinal_logits}

    def multi_loss(self, predictions, targets, *, horizon_mask=None,
                   event_pos_weight=None):
        """Supervise physical targets [B,4,4]; mask H=8-only expert records."""
        if targets.ndim != 3 or targets.shape[1:] != (4, 4):
            raise ValueError("Targets must be [B,4,4] physical units")
        if horizon_mask is None:
            horizon_mask = torch.ones(targets.shape[:2], device=targets.device)
        if horizon_mask.shape != targets.shape[:2]:
            raise ValueError("Invalid horizon mask")
        labels = (targets[..., :2, None] > self.dr_thresholds).to(targets.dtype)
        ord_loss = F.binary_cross_entropy_with_logits(
            predictions["multi_ordinal_logits"], labels,
            pos_weight=torch.as_tensor(self.ordinal_pos_weight, device=targets.device),
            reduction="none").mean(dim=(-1, -2))
        if event_pos_weight is None:
            event_pos_weight = targets.new_ones(2)
        evt_loss = F.binary_cross_entropy_with_logits(
            predictions["multi_event_logits"], targets[..., 2:],
            pos_weight=event_pos_weight, reduction="none").mean(dim=-1)
        mask = horizon_mask.to(targets.dtype)
        return ((ord_loss + evt_loss) * mask).sum() / mask.sum().clamp_min(1)
