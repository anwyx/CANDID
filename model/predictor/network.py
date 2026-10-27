"""MLP that maps a state window and an action chunk to disturbance labels."""
from __future__ import annotations

import torch
from torch import nn


class DisturbancePredictor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        target_dim: int = 4,
        obs_horizon: int = 2,
        action_horizon: int = 12,
        hidden: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.target_dim = target_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        width = obs_horizon * obs_dim + action_horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, target_dim),
        )

    def forward(self, obs, actions):
        batch = obs.shape[0]
        features = torch.cat((obs.reshape(batch, -1), actions.reshape(batch, -1)), dim=-1)
        return self.net(features)

    def loss(self, obs, actions, targets):
        return nn.functional.mse_loss(self.forward(obs, actions), targets)
