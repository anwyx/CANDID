import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.merge_datasets import merge_datasets
from candid.data.validate import validate_dataset


def make_collection(root: Path, marker: int) -> Path:
    (root / "scenes").mkdir(parents=True)
    (root / "disturbance_traces").mkdir()
    np.savez_compressed(root / "scenes/scene_000.npz", qpos=np.array([marker]))
    with (root / "trajectories.h5").open("wb"):
        pass
    with h5py.File(root / "trajectories.h5", "w") as handle:
        group = handle.create_group("traj_0")
        group.create_dataset("actions", data=np.full((14, 7), marker, dtype=np.float32))
        obs = group.create_group("obs")
        obs.create_dataset("state", data=np.zeros((15, 4), dtype=np.float32))
    n = 2
    np.savez_compressed(
        root / "disturbance_traces/traj_000.npz",
        distractor_poses=np.zeros((15, n, 7), dtype=np.float32),
        distractor_linear_velocities=np.zeros((15, n, 3), dtype=np.float32),
        distractor_angular_velocities=np.zeros((15, n, 3), dtype=np.float32),
        object_sizes_m=np.ones(n, dtype=np.float32),
    )
    scenes = [{"scene_id": "scene_000", "initial_state": "scenes/scene_000.npz"}]
    trajectories = [{
        "scene_id": "scene_000", "trajectory_id": "traj_000", "num_steps": 14,
        "object_ids": ["a", "b"], "saved_hdf5": True,
        "hdf5_file": "trajectories.h5", "hdf5_trajectory_key": "traj_0",
        "disturbance_trace": "disturbance_traces/traj_000.npz",
        "task_success": True, "control_mode": "pd_joint_pos", "obs_mode": "state_dict",
    }]
    for name, records in (("scenes.jsonl", scenes), ("trajectories.jsonl", trajectories)):
        (root / name).write_text("\n".join(json.dumps(x) for x in records) + "\n")
    return root


def test_merge_three_colliding_datasets(tmp_path):
    paths = [make_collection(tmp_path / f"source_{i}", i) for i in range(1, 4)]
    output = tmp_path / "merged"
    report = merge_datasets(paths, output)
    assert report == {"num_scenes": 3, "num_trajectories": 3, "num_recorded_hdf5": 3}
    assert validate_dataset(output) == report
    scenes = [json.loads(x) for x in (output / "scenes.jsonl").read_text().splitlines()]
    trajectories = [json.loads(x) for x in (output / "trajectories.jsonl").read_text().splitlines()]
    assert len({r["scene_id"] for r in scenes}) == 3
    assert len({r["trajectory_id"] for r in trajectories}) == 3
    assert {r["hdf5_file"] for r in trajectories} == {
        "team_01/trajectories.h5", "team_02/trajectories.h5", "team_03/trajectories.h5"
    }
    for i, row in enumerate(trajectories, 1):
        with h5py.File(output / row["hdf5_file"]) as handle:
            assert handle[row["hdf5_trajectory_key"]]["actions"][0, 0] == i
    assert all((p / "scenes.jsonl").exists() for p in paths)


def test_merge_symlink_and_existing_destination(tmp_path):
    a = make_collection(tmp_path / "a", 1)
    b = make_collection(tmp_path / "b", 2)
    dest = tmp_path / "out"
    merge_datasets([a, b], dest, mode="symlink")
    assert (dest / "team_01/trajectories.h5").is_symlink()
    assert validate_dataset(dest)["num_trajectories"] == 2
    with pytest.raises(FileExistsError):
        merge_datasets([a, b], dest)


def test_merge_invalid_source_does_not_create_output(tmp_path):
    a = make_collection(tmp_path / "a", 1)
    b = make_collection(tmp_path / "b", 2)
    (b / "disturbance_traces/traj_000.npz").unlink()
    dest = tmp_path / "merged"
    with pytest.raises(FileNotFoundError):
        merge_datasets([a, b], dest)
    assert not dest.exists()
