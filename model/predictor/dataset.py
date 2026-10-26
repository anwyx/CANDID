"""Predictor windows. Successes and failures are both kept. Test scenes are not."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from candid.data.adapter import observation_vector
from candid.data.windows import prepare_window, windows

TARGET_FIELDS = ("dr_mean", "dr_max", "topple_count", "fall_count")


def _target(labels: dict) -> np.ndarray:
    return np.asarray([float(labels.get(key, 0.0)) for key in TARGET_FIELDS], dtype=np.float32)


class PredictorWindowDataset(Dataset):
    def __init__(self, trajectories, scene_ids, obs_horizon: int = 2, action_horizon: int = 12):
        allowed = set(scene_ids)
        self.observation_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.samples = []
        for trajectory in trajectories:
            if trajectory.scene_id not in allowed:
                continue
            for time_index in windows(len(trajectory.actions), obs_horizon, action_horizon):
                window = prepare_window(
                    trajectory.observations,
                    trajectory.actions,
                    trajectory.trace,
                    time_index,
                    observation_horizon=obs_horizon,
                    action_horizon=action_horizon,
                )
                self.samples.append(
                    {
                        "obs": observation_vector(window["obs"]),
                        "action": np.asarray(window["actions"], dtype=np.float32),
                        "target": _target(window["predictor_labels"]),
                        "scene_id": trajectory.scene_id,
                        "task_success": trajectory.task_success,
                    }
                )
        if not self.samples:
            raise ValueError("Scene split produced no predictor windows")
        self.obs_dim = int(self.samples[0]["obs"].shape[-1])
        self.action_dim = int(self.samples[0]["action"].shape[-1])
        self.target_dim = int(self.samples[0]["target"].shape[-1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "obs": torch.from_numpy(sample["obs"]),
            "action": torch.from_numpy(sample["action"]),
            "target": torch.from_numpy(sample["target"]),
        }

    @property
    def successes(self):
        return [sample["task_success"] for sample in self.samples]
