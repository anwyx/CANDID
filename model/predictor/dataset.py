"""Predictor-only lazy trajectory windows; no changes to the shared data pipeline."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from candid.data.adapter import canonical_observation, observation_vector
from candid.data.metrics import disturbance_from_poses

TARGET_FIELDS = ("dr_mean", "dr_max", "any_topple", "any_fall")
EVENT_KEYS = ("distractor_toppled", "distractor_fallen")


class PredictorWindowDataset(Dataset):
    """Read one trajectory at a time and construct windows on demand.

    Only manifest rows and (trajectory index, timestep) pairs are stored eagerly.
    A bounded per-process LRU cache avoids reopening HDF5 files for every sample.
    Missing event labels are rejected by default, or masked when explicitly allowed.
    """

    def __init__(
        self, trajectories, scene_ids, obs_horizon=2, action_horizon=8,
        allow_missing_event_labels=False, cache_size=2,
    ):
        if obs_horizon < 1 or action_horizon < 1 or cache_size < 1:
            raise ValueError("Horizons and cache size must be positive")
        self.observation_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.allow_missing_event_labels = bool(allow_missing_event_labels)
        self.cache_size = int(cache_size)
        self._cache = OrderedDict()
        allowed = set(scene_ids)
        self.root = Path(trajectories) if isinstance(trajectories, (str, Path)) else None
        self.records = []
        self.skipped_short = 0
        self.missing_event_trajectories = 0
        self.missing_event_windows = 0
        index_pairs = []

        if self.root is not None:
            with (self.root / "trajectories.jsonl").open(encoding="utf-8") as handle:
                candidates = (json.loads(line) for line in handle if line.strip())
                for row in candidates:
                    if row["scene_id"] not in allowed or not row.get("saved_hdf5"):
                        continue
                    if row.get("control_mode", "pd_joint_pos") != "pd_joint_pos":
                        raise ValueError(f"{row['trajectory_id']}: expected pd_joint_pos")
                    if row.get("obs_mode", "state_dict") != "state_dict":
                        raise ValueError(f"{row['trajectory_id']}: expected state_dict")
                    length = int(row["num_steps"])
                    count = length - self.observation_horizon - self.action_horizon + 2
                    if count <= 0:
                        self.skipped_short += 1
                        continue
                    with np.load(self.root / row["disturbance_trace"], allow_pickle=False) as archive:
                        missing = [key for key in EVENT_KEYS if key not in archive.files]
                    if missing and not self.allow_missing_event_labels:
                        raise ValueError(
                            f"{row['trajectory_id']} missing {missing}; "
                            "use --allow-missing-event-labels to mask unknown events"
                        )
                    if missing:
                        self.missing_event_trajectories += 1
                        self.missing_event_windows += count
                    trajectory_index = len(self.records)
                    self.records.append(row)
                    index_pairs.extend(
                        (trajectory_index, t)
                        for t in range(self.observation_horizon - 1,
                                       length - self.action_horizon + 1)
                    )
        else:
            for trajectory in trajectories:
                if trajectory.scene_id not in allowed:
                    continue
                length = len(trajectory.actions)
                count = length - self.observation_horizon - self.action_horizon + 2
                if count <= 0:
                    self.skipped_short += 1
                    continue
                missing = [key for key in EVENT_KEYS if key not in trajectory.trace]
                if missing and not self.allow_missing_event_labels:
                    raise ValueError(
                        f"{trajectory.trajectory_id} missing {missing}; "
                        "use allow_missing_event_labels=True to mask unknown events"
                    )
                if missing:
                    self.missing_event_trajectories += 1
                    self.missing_event_windows += count
                trajectory_index = len(self.records)
                self.records.append(trajectory)
                index_pairs.extend(
                    (trajectory_index, t)
                    for t in range(self.observation_horizon - 1,
                                   length - self.action_horizon + 1)
                )

        if not index_pairs:
            raise ValueError("Scene split produced no predictor windows")
        self.indices = np.asarray(index_pairs, dtype=np.int32)
        first = self._sequence(0)
        self.obs_dim = int(first["obs"].shape[-1])
        self.action_dim = int(first["actions"].shape[-1])
        self.target_dim = len(TARGET_FIELDS)

    def __len__(self):
        return len(self.indices)

    def _sequence(self, trajectory_index):
        if trajectory_index in self._cache:
            self._cache.move_to_end(trajectory_index)
            return self._cache[trajectory_index]

        record = self.records[trajectory_index]
        if self.root is None:
            observations = observation_vector(record.observations)
            actions = np.asarray(record.actions, dtype=np.float32)
            trace = record.trace
        else:
            file_path = self.root / record["hdf5_file"]
            with h5py.File(file_path, "r") as handle:
                group = handle[record["hdf5_trajectory_key"]]
                actions = np.asarray(group["actions"], dtype=np.float32)
                observations = observation_vector(
                    canonical_observation(group["obs"], len(record["object_ids"]))
                )
            with np.load(self.root / record["disturbance_trace"], allow_pickle=False) as archive:
                trace = {
                    "distractor_poses": archive["distractor_poses"].copy(),
                    "object_sizes_m": archive["object_sizes_m"].copy(),
                }
                for key in EVENT_KEYS:
                    if key in archive.files:
                        trace[key] = archive[key].copy()
        if len(observations) != len(actions) + 1:
            raise ValueError("Predictor observations/actions are not T/T+1 aligned")
        if len(trace["distractor_poses"]) != len(actions) + 1:
            raise ValueError("Predictor physics trace/actions are not T/T+1 aligned")
        if not np.isfinite(observations).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite predictor observations/actions")
        entry = {"obs": observations, "actions": actions, "trace": trace}
        self._cache[trajectory_index] = entry
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return entry

    def __getitem__(self, index):
        trajectory_index, t = (int(v) for v in self.indices[index])
        entry = self._sequence(trajectory_index)
        horizon = self.action_horizon
        obs = entry["obs"][t - self.observation_horizon + 1:t + 1]
        action = entry["actions"][t:t + horizon]
        trace = entry["trace"]
        labels = disturbance_from_poses(
            trace["distractor_poses"], trace["object_sizes_m"],
            start=t, stop=t + horizon,
            topple_flags=trace.get(EVENT_KEYS[0]),
            fall_flags=trace.get(EVENT_KEYS[1]),
        )
        event_mask = np.asarray([
            float("topple_count" in labels), float("fall_count" in labels)
        ], dtype=np.float32)
        target = np.asarray([
            labels["dr_mean"], labels["dr_max"],
            float(labels.get("topple_count", 0) > 0),
            float(labels.get("fall_count", 0) > 0),
        ], dtype=np.float32)
        return {
            "obs": torch.from_numpy(np.array(obs, dtype=np.float32, copy=True)),
            "action": torch.from_numpy(np.array(action, dtype=np.float32, copy=True)),
            "target": torch.from_numpy(target),
            "event_mask": torch.from_numpy(event_mask),
        }

    def metadata(self, index):
        trajectory_index, t = (int(v) for v in self.indices[index])
        record = self.records[trajectory_index]
        if self.root is None:
            return record.scene_id, record.trajectory_id, t
        return record["scene_id"], record["trajectory_id"], t

    @property
    def successes(self):
        result = []
        for trajectory_index, _ in self.indices:
            record = self.records[int(trajectory_index)]
            result.append(
                record.task_success if self.root is None else bool(record["task_success"])
            )
        return result



class TrajectoryBatchSampler:
    """Shuffle trajectory-local batches to avoid random full-HDF5 reload per sample."""

    def __init__(self, dataset, batch_size, seed=0):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.groups = []
        indices = dataset.indices[:, 0]
        for trajectory_index in range(len(dataset.records)):
            positions = np.flatnonzero(indices == trajectory_index)
            if len(positions):
                self.groups.append(positions)

    def __len__(self):
        return sum((len(group) + self.batch_size - 1) // self.batch_size
                   for group in self.groups)

    def __iter__(self):
        batches = []
        for positions in self.groups:
            order = torch.randperm(len(positions), generator=self.generator).numpy()
            shuffled = positions[order]
            for start in range(0, len(shuffled), self.batch_size):
                batches.append(shuffled[start:start + self.batch_size].tolist())
        order = torch.randperm(len(batches), generator=self.generator).tolist()
        for index in order:
            yield batches[index]
