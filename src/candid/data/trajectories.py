"""Read recorded CANDID trajectories. No model-specific windowing or networks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from candid.data.adapter import canonical_observation


@dataclass
class Trajectory:
    scene_id: str
    trajectory_id: str
    task_success: bool
    observations: dict
    actions: np.ndarray
    trace: dict


def _rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _copy_group(group, key):
    return np.asarray(group[key], dtype=np.float32).copy()


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
