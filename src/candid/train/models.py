"""Small state-conditioned diffusion policy and disturbance predictor.

These are pilot networks for the fixed CANDID window format. They train from
saved state and action arrays, without creating a ManiSkill environment.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _timestep_embedding(timesteps, dim):
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class DiffusionPolicy(nn.Module):
    """Predict a chunk of ``pd_joint_pos`` actions from a state history.

    The action tensor is normalized by the caller. ``sample`` returns noise
    predictions in that same normalized space.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int = 2,
        action_horizon: int = 12,
        diffusion_steps: int = 8,
        hidden: int = 256,
        time_dim: int = 16,
    ):
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.diffusion_steps = diffusion_steps
        self.time_dim = time_dim
        chunk = action_horizon * action_dim
        condition = obs_horizon * obs_dim
        self.net = nn.Sequential(
            nn.Linear(chunk + condition + time_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, chunk),
        )
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))

    def _noise(self, noisy_action, obs, timesteps):
        batch = noisy_action.shape[0]
        features = torch.cat(
            (
                noisy_action.reshape(batch, -1),
                obs.reshape(batch, -1),
                _timestep_embedding(timesteps, self.time_dim),
            ),
            dim=-1,
        )
        return self.net(features).view_as(noisy_action)

    def loss(self, obs, actions):
        batch = actions.shape[0]
        timesteps = torch.randint(0, self.diffusion_steps, (batch,), device=actions.device)
        noise = torch.randn_like(actions)
        scale = self.alpha_bar[timesteps].view(batch, 1, 1)
        noisy = scale.sqrt() * actions + (1.0 - scale).sqrt() * noise
        return nn.functional.mse_loss(self._noise(noisy, obs, timesteps), noise)

    @torch.no_grad()
    def sample(self, obs):
        batch = obs.shape[0]
        action = torch.randn(
            batch, self.action_horizon, self.action_dim, device=obs.device
        )
        for index in reversed(range(self.diffusion_steps)):
            timesteps = torch.full((batch,), index, device=obs.device, dtype=torch.long)
            predicted = self._noise(action, obs, timesteps)
            beta = self.betas[index]
            alpha = self.alphas[index]
            bar = self.alpha_bar[index]
            mean = (1.0 / alpha.sqrt()) * (action - beta / (1.0 - bar).sqrt() * predicted)
            if index > 0:
                action = mean + beta.sqrt() * torch.randn_like(action)
            else:
                action = mean
        return action


class DisturbancePredictor(nn.Module):
    """Map a state history and an action chunk to chunk-level consequences."""

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
