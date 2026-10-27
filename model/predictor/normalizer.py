"""Mean/std normalizer for predictor inputs and the four disturbance labels."""
from __future__ import annotations

import numpy as np
import torch


class PredictorNormalizer:
    def __init__(self, obs_mean, obs_std, action_mean, action_std, target_mean, target_std):
        self.obs_mean = torch.as_tensor(obs_mean, dtype=torch.float32)
        self.obs_std = torch.as_tensor(obs_std, dtype=torch.float32)
        self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.as_tensor(action_std, dtype=torch.float32)
        self.target_mean = torch.as_tensor(target_mean, dtype=torch.float32)
        self.target_std = torch.as_tensor(target_std, dtype=torch.float32)

    def to(self, device):
        return PredictorNormalizer(
            self.obs_mean.to(device),
            self.obs_std.to(device),
            self.action_mean.to(device),
            self.action_std.to(device),
            self.target_mean.to(device),
            self.target_std.to(device),
        )

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def normalize_action(self, action):
        return (action - self.action_mean) / self.action_std

    def normalize_target(self, target):
        return (target - self.target_mean) / self.target_std

    def denormalize_target(self, target):
        return target * self.target_std + self.target_mean


def fit_predictor_normalizer(dataset) -> PredictorNormalizer:
    obs = np.stack([sample["obs"] for sample in dataset.samples]).astype(np.float32)
    action = np.stack([sample["action"] for sample in dataset.samples]).astype(np.float32)
    target = np.stack([sample["target"] for sample in dataset.samples]).astype(np.float32)
    return PredictorNormalizer(
        obs.mean(axis=(0, 1)),
        np.maximum(obs.std(axis=(0, 1)), 1e-6),
        action.mean(axis=(0, 1)),
        np.maximum(action.std(axis=(0, 1)), 1e-6),
        target.mean(axis=0),
        np.maximum(target.std(axis=0), 1e-6),
    )
