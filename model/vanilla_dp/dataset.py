"""CANDID trajectories sliced the way ManiSkill's Diffusion Policy dataset slices them.

The index arithmetic is the ``SmallDemoDataset_DiffusionPolicy`` loop in
ManiSkill ``examples/baselines/diffusion_policy/train.py`` at commit
62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3:

    pad_before = obs_horizon - 1
    pad_after = pred_horizon - obs_horizon
    starts in [-pad_before, L - pred_horizon + pad_after)

Samples before the trajectory repeat the first observation and action.
Samples past the end repeat the last absolute joint target. Upstream instead
zero-pads the arm and copies the gripper, and it only defines that pad for
delta control. A zero arm command is not "stay still" for ``pd_joint_pos``.

Only successful trajectories from the frozen train scenes are included.
Upstream reads every demo in a ManiSkill HDF5 and has no scene split.
Observations stay in the canonical CANDID vector; they are not normalized.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from candid.data.adapter import observation_vector


class DiffusionWindowDataset(Dataset):
    def __init__(self, trajectories, scene_ids, obs_horizon: int, pred_horizon: int):
        allowed = set(scene_ids)
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.sequences = []
        for trajectory in trajectories:
            if trajectory.scene_id not in allowed or not trajectory.task_success:
                continue
            observations = observation_vector(trajectory.observations)
            actions = np.asarray(trajectory.actions, dtype=np.float32)
            if observations.shape[0] != actions.shape[0] + 1:
                raise ValueError(f"{trajectory.trajectory_id} is not T / T+1 aligned")
            if not np.isfinite(observations).all() or not np.isfinite(actions).all():
                raise ValueError(f"{trajectory.trajectory_id} has non-finite data")
            self.sequences.append((observations, actions))
        if not self.sequences:
            raise ValueError("No successful training trajectories for Vanilla DP")
        self.obs_dim = int(self.sequences[0][0].shape[-1])
        self.action_dim = int(self.sequences[0][1].shape[-1])
        if any(actions.shape[-1] != self.action_dim for _, actions in self.sequences):
            raise ValueError("Mixed action dimensions in one dataset")
        pad_before = self.obs_horizon - 1
        pad_after = self.pred_horizon - self.obs_horizon
        self.slices = []
        for index, (_, actions) in enumerate(self.sequences):
            length = actions.shape[0]
            self.slices.extend(
                (index, start, start + self.pred_horizon)
                for start in range(-pad_before, length - self.pred_horizon + pad_after)
            )
        if not self.slices:
            raise ValueError("Trajectories produced no diffusion-policy windows")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        trajectory_index, start, end = self.slices[index]
        observations, actions = self.sequences[trajectory_index]
        length = actions.shape[0]
        obs_seq = observations[max(0, start) : start + self.obs_horizon]
        act_seq = actions[max(0, start) : end]
        if start < 0:
            obs_seq = np.concatenate([np.repeat(obs_seq[:1], -start, axis=0), obs_seq], axis=0)
            act_seq = np.concatenate([np.repeat(act_seq[:1], -start, axis=0), act_seq], axis=0)
        if end > length:
            act_seq = np.concatenate(
                [act_seq, np.repeat(act_seq[-1:], end - length, axis=0)], axis=0
            )
        if obs_seq.shape[0] != self.obs_horizon or act_seq.shape[0] != self.pred_horizon:
            raise RuntimeError("Diffusion window padding did not reach the required horizon")
        return {
            "observations": torch.from_numpy(np.ascontiguousarray(obs_seq)),
            "actions": torch.from_numpy(np.ascontiguousarray(act_seq)),
        }

    def training_actions(self) -> np.ndarray:
        return np.concatenate([actions for _, actions in self.sequences], axis=0)
