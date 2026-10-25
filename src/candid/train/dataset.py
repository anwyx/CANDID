"""Training windows for the diffusion policy and the disturbance predictor.

Reads a validated collection directory. Observations come from
``canonical_observation`` and chunks from ``prepare_window``. Future object
poses stay inside the predictor labels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from candid.data.adapter import canonical_observation
from candid.data.windows import prepare_window, windows

OBSERVATION_HORIZON = 2
ACTION_HORIZON = 12
OBSERVATION_FIELDS = (
    "robot_qpos",
    "robot_qvel",
    "tcp_pose",
    "target_state",
    "distractor_states",
    "distractor_mask",
    "goal_xy",
)
PREDICTOR_TARGET_FIELDS = ("dr_mean", "dr_max", "topple_count", "fall_count")


@dataclass
class Trajectory:
    scene_id: str
    trajectory_id: str
    task_success: bool
    observations: dict
    actions: np.ndarray
    trace: dict


@dataclass
class Normalizer:
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor

    def to(self, device):
        return Normalizer(
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

    def denormalize_action(self, action):
        return action * self.action_std + self.action_mean

    def normalize_target(self, target):
        return (target - self.target_mean) / self.target_std

    def denormalize_target(self, target):
        return target * self.target_std + self.target_mean

    def as_payload(self):
        return {
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_payload(cls, payload):
        return cls(
            payload["obs_mean"],
            payload["obs_std"],
            payload["action_mean"],
            payload["action_std"],
            payload["target_mean"],
            payload["target_std"],
        )


def _rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _copy_group(group, key):
    return np.asarray(group[key], dtype=np.float32).copy()


def observation_vector(obs_window: dict) -> np.ndarray:
    """Stack one window's named state fields into [horizon, features]."""
    parts = []
    horizon = None
    for key in OBSERVATION_FIELDS:
        array = np.asarray(obs_window[key], dtype=np.float32)
        if horizon is None:
            horizon = array.shape[0]
        if array.shape[0] != horizon:
            raise ValueError(f"Observation field {key} has a different horizon")
        parts.append(array.reshape(horizon, -1))
    return np.concatenate(parts, axis=-1)


def predictor_target(labels: dict) -> np.ndarray:
    values = []
    for key in PREDICTOR_TARGET_FIELDS:
        values.append(float(labels.get(key, 0.0)))
    return np.asarray(values, dtype=np.float32)


def load_trajectories(root) -> list[Trajectory]:
    """Load every saved trajectory whose HDF5 and pose trace agree."""
    root = Path(root)
    rows = _rows(root / "trajectories.jsonl")
    loaded = []
    handles = {}
    try:
        for row in rows:
            if not row.get("saved_hdf5"):
                continue
            control_mode = row.get("control_mode")
            if control_mode is not None and control_mode != "pd_joint_pos":
                raise ValueError(
                    f"{row['trajectory_id']} uses {control_mode}, expected pd_joint_pos"
                )
            obs_mode = row.get("obs_mode")
            if obs_mode is not None and obs_mode != "state_dict":
                raise ValueError(
                    f"{row['trajectory_id']} uses {obs_mode}, expected state_dict"
                )
            trace_path = root / row["disturbance_trace"]
            with np.load(trace_path, allow_pickle=False) as archive:
                poses = archive["distractor_poses"].copy()
                sizes = archive["object_sizes_m"].copy()
                trace = {"distractor_poses": poses, "object_sizes_m": sizes}
                for key in ("distractor_toppled", "distractor_fallen"):
                    if key in archive:
                        trace[key] = archive[key].copy()
            filename = row["hdf5_file"]
            if filename not in handles:
                handles[filename] = h5py.File(root / filename, "r")
            group = handles[filename][row["hdf5_trajectory_key"]]
            actions = _copy_group(group, "actions")
            num_distractors = len(row["object_ids"])
            if poses.shape[1] != num_distractors:
                raise ValueError(f"Distractor count mismatch in {row['trajectory_id']}")
            if actions.shape[0] != poses.shape[0] - 1:
                raise ValueError(f"Action/trace length mismatch in {row['trajectory_id']}")
            observations = canonical_observation(group["obs"], num_distractors)
            loaded.append(
                Trajectory(
                    scene_id=row["scene_id"],
                    trajectory_id=row["trajectory_id"],
                    task_success=bool(row["task_success"]),
                    observations=observations,
                    actions=actions,
                    trace=trace,
                )
            )
    finally:
        for handle in handles.values():
            handle.close()
    if not loaded:
        raise ValueError(f"No trainable trajectories in {root}")
    return loaded


def split_scene_ids(scene_ids, val_ratio=0.2, seed=0):
    """Hold out whole scenes. One scene stays entirely in training."""
    ids = sorted(set(scene_ids))
    if len(ids) <= 1:
        return ids, []
    order = np.random.RandomState(seed).permutation(len(ids))
    ordered = [ids[index] for index in order]
    n_val = int(round(len(ids) * val_ratio))
    n_val = min(max(n_val, 1), len(ids) - 1)
    return ordered[n_val:], ordered[:n_val]


class WindowDataset(Dataset):
    """Sliding windows from a fixed set of scenes.

    ``success_only=True`` is the diffusion-policy set. The predictor set keeps
    failed attempts as well.
    """

    def __init__(
        self,
        trajectories: list[Trajectory],
        scene_ids,
        success_only: bool,
        observation_horizon: int = OBSERVATION_HORIZON,
        action_horizon: int = ACTION_HORIZON,
    ):
        allowed = set(scene_ids)
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.samples = []
        action_dim = None
        for trajectory in trajectories:
            if trajectory.scene_id not in allowed:
                continue
            if success_only and not trajectory.task_success:
                continue
            num_actions = len(trajectory.actions)
            if action_dim is None:
                action_dim = int(trajectory.actions.shape[-1])
            elif trajectory.actions.shape[-1] != action_dim:
                raise ValueError("Mixed action dimensions in one dataset")
            for time_index in windows(num_actions, observation_horizon, action_horizon):
                window = prepare_window(
                    trajectory.observations,
                    trajectory.actions,
                    trajectory.trace,
                    time_index,
                    observation_horizon=observation_horizon,
                    action_horizon=action_horizon,
                )
                self.samples.append(
                    {
                        "obs": observation_vector(window["obs"]),
                        "action": np.asarray(window["actions"], dtype=np.float32),
                        "target": predictor_target(window["predictor_labels"]),
                        "scene_id": trajectory.scene_id,
                        "task_success": trajectory.task_success,
                        "trajectory_id": trajectory.trajectory_id,
                    }
                )
        if not self.samples:
            raise ValueError("Scene split produced no training windows")
        self.action_dim = int(self.samples[0]["action"].shape[-1])
        self.obs_dim = int(self.samples[0]["obs"].shape[-1])

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
    def scene_ids(self):
        return {sample["scene_id"] for sample in self.samples}

    @property
    def successes(self):
        return [sample["task_success"] for sample in self.samples]


def fit_normalizer(dataset: WindowDataset, include_targets: bool) -> Normalizer:
    obs = np.stack([sample["obs"] for sample in dataset.samples]).astype(np.float32)
    action = np.stack([sample["action"] for sample in dataset.samples]).astype(np.float32)
    obs_mean = obs.mean(axis=(0, 1))
    obs_std = obs.std(axis=(0, 1))
    action_mean = action.mean(axis=(0, 1))
    action_std = action.std(axis=(0, 1))
    if include_targets:
        target = np.stack([sample["target"] for sample in dataset.samples]).astype(np.float32)
        target_mean = target.mean(axis=0)
        target_std = target.std(axis=0)
    else:
        target_mean = np.zeros(len(PREDICTOR_TARGET_FIELDS), dtype=np.float32)
        target_std = np.ones(len(PREDICTOR_TARGET_FIELDS), dtype=np.float32)
    return Normalizer(
        torch.from_numpy(obs_mean),
        torch.from_numpy(np.maximum(obs_std, 1e-6)),
        torch.from_numpy(action_mean),
        torch.from_numpy(np.maximum(action_std, 1e-6)),
        torch.from_numpy(target_mean.astype(np.float32)),
        torch.from_numpy(np.maximum(target_std, 1e-6).astype(np.float32)),
    )
