"""Trajectory-local 50/50 batch sampler for Expert + policy DP windows.

Expert trajectories live in HDF5 and policy trajectories in NPZ. Purely
random example-level sampling constantly reopens files; grouping by selected
trajectory within a batch makes the training path much cheaper on servers.
"""
from __future__ import annotations

import math

import numpy as np
from torch.utils.data import Sampler


class MixedTrajectoryBatchSampler(Sampler):
    def __init__(self, expert, policy, batch_size=128, seed=0):
        if batch_size < 2:
            raise ValueError("Mixed sampler needs batch_size >= 2")
        self.expert = self._groups(expert.indices[:, 0], offset=0)
        self.policy = self._groups(
            np.asarray(policy.indices, dtype=np.int64)[:, 0],
            offset=len(expert),
        )
        if not self.expert or not self.policy:
            raise ValueError("Both expert and policy sources need windows")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.iteration = 0
        self.steps = math.ceil(2 * max(len(expert), len(policy)) / batch_size)
        self.expert_weights = np.asarray([len(g) for g in self.expert], np.float64)
        self.expert_weights /= self.expert_weights.sum()
        self.policy_weights = np.asarray([len(g) for g in self.policy], np.float64)
        self.policy_weights /= self.policy_weights.sum()

    @staticmethod
    def _groups(trajectory_ids, offset):
        indices = np.arange(len(trajectory_ids), dtype=np.int64) + offset
        groups = []
        for trajectory in np.unique(trajectory_ids):
            groups.append(indices[trajectory_ids == trajectory])
        return groups

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.iteration)
        self.iteration += 1
        n_expert = self.batch_size // 2
        n_policy = self.batch_size - n_expert
        for _ in range(self.steps):
            a = self.expert[rng.choice(len(self.expert), p=self.expert_weights)]
            b = self.policy[rng.choice(len(self.policy), p=self.policy_weights)]
            indices = np.concatenate([
                rng.choice(a, size=n_expert, replace=len(a) < n_expert),
                rng.choice(b, size=n_policy, replace=len(b) < n_policy),
            ])
            rng.shuffle(indices)
            yield indices.tolist()
