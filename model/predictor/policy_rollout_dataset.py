"""Lazy supervised windows from DP closed-loop rollouts (train scenes only).

The underlying NPZ records canonical observations [T+1,O], executed joint
targets [T,A] and object poses [T+1,N,7]. All labels are local ground truth,
never episode-wide DR. Files are opened lazily to bound resident memory.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from candid.data.metrics import disturbance_from_poses


class PolicyRolloutWindowDataset(Dataset):
    def __init__(self, run_dir, allowed_scenes, *, obs_horizon=2,
                 action_horizon=8, label_horizon=4, cache_size=2):
        self.root = Path(run_dir)
        if not (1 <= label_horizon <= action_horizon):
            raise ValueError("label_horizon must be in [1, action_horizon]")
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.label_horizon = int(label_horizon)
        self._cache = OrderedDict()
        self.cache_size = int(cache_size)
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        if summary.get("split_name") != "train":
            raise ValueError("Policy rollouts MUST be generated with --split train")
        allowed = set(allowed_scenes)
        rows = [
            json.loads(line)
            for line in (self.root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError("No policy rollouts recorded")
        unapproved = {row["scene_id"] for row in rows} - allowed
        if unapproved:
            raise ValueError(f"Policy rollouts include non-training scenes: {sorted(unapproved)}")
        self.records = []
        self.indices = []
        self.obs_dim = self.action_dim = None
        for row in rows:
            path = Path(row["trajectory_file"])
            if not path.is_file():
                path = self.root / path.name if (self.root / path.name).is_file() else self.root / path
            if not path.is_file():
                # Use episode names from the manifest; never silently load
                # another dataset or reconstruct missing observations.
                raise FileNotFoundError(f"Missing policy rollout: {row['trajectory_file']}")
            with np.load(path, allow_pickle=False) as data:
                necessary = {"observations", "actions", "distractor_poses",
                             "object_sizes_m", "distractor_toppled", "distractor_fallen"}
                missing = necessary - set(data.files)
                if missing:
                    raise ValueError(
                        f"{path} lacks rollout supervision {sorted(missing)}; "
                        "recollect using updated eval_dp.py --split train"
                    )
                obs = data["observations"]
                actions = data["actions"]
                poses = data["distractor_poses"]
                if len(obs) != len(actions) + 1 or len(poses) != len(obs):
                    raise ValueError(f"Unaligned rollout in {path}")
                if self.obs_dim is None:
                    self.obs_dim, self.action_dim = obs.shape[-1], actions.shape[-1]
                elif (obs.shape[-1], actions.shape[-1]) != (self.obs_dim, self.action_dim):
                    raise ValueError(f"Rollout feature dimension mismatch: {path}")
                length = len(actions)
            record_id = len(self.records)
            self.records.append(path)
            for t in range(self.obs_horizon-1, length-self.action_horizon+1):
                self.indices.append((record_id, t))
        if not self.indices:
            raise ValueError("No usable rollout windows")

    def __len__(self):
        return len(self.indices)

    def _load(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        with np.load(self.records[index], allow_pickle=False) as data:
            entry = {key: data[key].copy() for key in (
                "observations", "actions", "distractor_poses",
                "object_sizes_m", "distractor_toppled", "distractor_fallen"
            )}
        self._cache[index] = entry
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return entry

    def __getitem__(self, index):
        trajectory, t = self.indices[index]
        x = self._load(trajectory)
        H = self.action_horizon
        labels = disturbance_from_poses(
            x["distractor_poses"], x["object_sizes_m"],
            start=t, stop=t + self.label_horizon,
            topple_flags=x["distractor_toppled"],
            fall_flags=x["distractor_fallen"],
        )
        return {
            "obs": torch.as_tensor(
                x["observations"][t-self.obs_horizon+1:t+1].copy(),
                dtype=torch.float32
            ),
            "action": torch.as_tensor(x["actions"][t:t+H].copy(), dtype=torch.float32),
            "target": torch.tensor([
                labels["dr_mean"], labels["dr_max"],
                float(labels["any_topple"]), float(labels["any_fall"])
            ], dtype=torch.float32),
            "event_mask": torch.ones(2, dtype=torch.float32),
        }
