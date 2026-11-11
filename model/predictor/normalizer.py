"""Streaming normalization for predictor DR regression and action/state inputs."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.predictor.action_features import absolute_to_delta_q


class PredictorNormalizer:
    def __init__(self, obs_mean, obs_std, action_mean, action_std,
                 target_mean, target_std, event_pos_weight=None,
                 action_representation="absolute"):
        if action_representation not in ("absolute", "delta_q"):
            raise ValueError("Unknown predictor action representation")
        self.action_representation = action_representation
        self.obs_mean = torch.as_tensor(obs_mean, dtype=torch.float32)
        self.obs_std = torch.as_tensor(obs_std, dtype=torch.float32)
        self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.as_tensor(action_std, dtype=torch.float32)
        self.target_mean = torch.as_tensor(target_mean, dtype=torch.float32)
        self.target_std = torch.as_tensor(target_std, dtype=torch.float32)
        self.event_pos_weight = torch.as_tensor(
            [1.0, 1.0] if event_pos_weight is None else event_pos_weight,
            dtype=torch.float32,
        )

    def to(self, device):
        return PredictorNormalizer(
            self.obs_mean.to(device), self.obs_std.to(device),
            self.action_mean.to(device), self.action_std.to(device),
            self.target_mean.to(device), self.target_std.to(device),
            self.event_pos_weight.to(device),
            action_representation=self.action_representation,
        )

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def normalize_action(self, action, obs=None):
        if self.action_representation == "delta_q":
            if obs is None:
                raise ValueError("delta_q normalization requires raw observation")
            action = absolute_to_delta_q(obs, action)
        return (action - self.action_mean) / self.action_std

    def normalize_target(self, target):
        """Only DR values are normalized. Binary events remain 0/1."""
        return torch.cat((
            (target[..., :2] - self.target_mean) / self.target_std,
            target[..., 2:],
        ), dim=-1)

    def denormalize_target(self, target):
        return target * self.target_std + self.target_mean


def normalize_predictor_actions(normalizer, actions, *, obs):
    """Use raw observations only for delta-q; preserve legacy normalizer API.

    The legacy absolute-action normalizer (including test IdentityNormalizer)
    accepts normalize_action(actions) without an obs argument. Scene1 v3
    delta-q needs raw obs before normalization and remains differentiable.
    """
    representation = getattr(normalizer, "action_representation", "absolute")
    if representation == "delta_q":
        return normalizer.normalize_action(actions, obs=obs)
    if representation == "absolute":
        return normalizer.normalize_action(actions)
    raise ValueError(f"Unknown predictor action representation: {representation}")


def fit_predictor_normalizer(dataset, batch_size=256, *
                             action_representation="absolute") -> PredictorNormalizer:
    """One streaming pass; no stack of all overlapping predictor windows."""
    totals = {}
    count = {}
    positives = np.zeros(2, dtype=np.float64)
    known = np.zeros(2, dtype=np.float64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        actions = batch["action"]
        if action_representation == "delta_q":
            actions = absolute_to_delta_q(batch["obs"], actions)
        for name, values, axes in (
            ("obs", batch["obs"], (0, 1)),
            ("action", actions, (0, 1)),
            ("target", batch["target"][:, :2], (0,)),
        ):
            data = values.numpy().astype(np.float64)
            n = np.prod([data.shape[axis] for axis in axes])
            summed = data.sum(axis=axes)
            squared = np.square(data).sum(axis=axes)
            if name not in totals:
                totals[name] = [np.zeros_like(summed), np.zeros_like(summed)]
                count[name] = 0
            totals[name][0] += summed
            totals[name][1] += squared
            count[name] += n
        event = batch["target"][:, 2:].numpy()
        mask = batch["event_mask"].numpy()
        positives += (event * mask).sum(axis=0)
        known += mask.sum(axis=0)

    def stats(name):
        mean = totals[name][0] / count[name]
        var = totals[name][1] / count[name] - mean ** 2
        return mean.astype(np.float32), np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)

    obs_mean, obs_std = stats("obs")
    action_mean, action_std = stats("action")
    target_mean, target_std = stats("target")
    # A finite cap prevents very rare events from destabilizing optimization.
    weights = np.clip((known - positives) / np.maximum(positives, 1), 1.0, 50.0)
    return PredictorNormalizer(
        obs_mean, obs_std, action_mean, action_std,
        target_mean, target_std, weights.astype(np.float32),
        action_representation=action_representation,
    )
