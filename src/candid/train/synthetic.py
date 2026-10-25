"""Write a tiny collection with the same files a real ``collect.py`` run produces.

The trajectories are numeric stand-ins. They let the training pipeline be
executed before teammates finish generating simulation data. Action width is
7, matching xArm6 ``pd_joint_pos`` (six arm joints and one gripper command).
Joint state width is 12, matching the simulated robot state rather than the
command.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from candid.data.metrics import disturbance_from_poses, event_flags_from_poses
from candid.data.storage import DatasetWriter
from candid.data.validate import validate_dataset

ACTION_DIM = 7
QPOS_DIM = 12
NUM_DISTRACTORS = 4
NUM_ACTIONS = 20


class _Trace:
    def __init__(self, arrays, names):
        self._arrays = arrays
        self.names = names
        self.steps = arrays["distractor_poses"].shape[0] - 1
        self.sizes = arrays["object_sizes_m"]

    def arrays(self):
        return self._arrays

    def summary(self):
        return disturbance_from_poses(
            self._arrays["distractor_poses"],
            self.sizes,
            topple_flags=self._arrays["distractor_toppled"],
            fall_flags=self._arrays["distractor_fallen"],
        )


def _episode(seed: int, shift: float):
    rng = np.random.RandomState(seed)
    frames = NUM_ACTIONS + 1
    time = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    qpos = 0.2 * np.sin(time[:, None] + rng.uniform(0, 1, size=(1, QPOS_DIM)))
    qvel = np.gradient(qpos, axis=0).astype(np.float32)
    tcp = np.zeros((frames, 7), dtype=np.float32)
    tcp[:, 0] = 0.05 * time
    tcp[:, 3] = 1.0
    target_pose = np.zeros((frames, 7), dtype=np.float32)
    target_pose[:, 0] = -0.12 + shift * time
    target_pose[:, 3] = 1.0
    target_linear = np.zeros((frames, 3), dtype=np.float32)
    target_linear[:, 0] = shift / NUM_ACTIONS
    target_angular = np.zeros((frames, 3), dtype=np.float32)
    target_dimensions = np.full((frames, 3), 0.04, dtype=np.float32)
    poses = np.zeros((frames, NUM_DISTRACTORS, 7), dtype=np.float32)
    poses[:, :, 3] = 1.0
    poses[:, 0, 0] = shift * time
    poses[:, 1, 1] = 0.5 * shift * time
    flags = event_flags_from_poses(poses, kinds=["box"] * NUM_DISTRACTORS)
    actions = np.zeros((NUM_ACTIONS, ACTION_DIM), dtype=np.float32)
    actions[:, :] = qpos[:-1, :ACTION_DIM]
    actions[:, -1] = np.where(time[:-1] < 0.7, 0.8, 0.0)
    return {
        "qpos": qpos.astype(np.float32),
        "qvel": qvel,
        "tcp_pose": tcp,
        "target_pose": target_pose,
        "target_linear_velocity": target_linear,
        "target_angular_velocity": target_angular,
        "target_dimensions": target_dimensions,
        "distractor_poses": poses.reshape(frames, -1),
        "distractor_linear_velocities": np.zeros((frames, NUM_DISTRACTORS * 3), dtype=np.float32),
        "distractor_angular_velocities": np.zeros((frames, NUM_DISTRACTORS * 3), dtype=np.float32),
        "distractor_dimensions": np.full((frames, NUM_DISTRACTORS * 3), 0.04, dtype=np.float32),
        "goal_pos": np.tile(np.array([0.10, 0.0, 0.001], dtype=np.float32), (frames, 1)),
        "actions": actions,
        "trace_poses": poses,
        "trace_flags": flags,
    }


def _write_obs(group, episode):
    agent = group.create_group("agent")
    extra = group.create_group("extra")
    for key in ("qpos", "qvel"):
        agent.create_dataset(key, data=episode[key])
    for key in (
        "tcp_pose",
        "target_pose",
        "target_linear_velocity",
        "target_angular_velocity",
        "target_dimensions",
        "distractor_poses",
        "distractor_linear_velocities",
        "distractor_angular_velocities",
        "distractor_dimensions",
        "goal_pos",
    ):
        extra.create_dataset(key, data=episode[key])


def write_synthetic_dataset(root, num_scenes: int = 3) -> Path:
    """Create scenes, trajectories, one HDF5 file, and pose traces."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    writer = DatasetWriter(root)
    h5_name = "trajectories_synthetic.h5"
    h5_path = root / h5_name
    episode_index = 0
    with h5py.File(h5_path, "w") as handle:
        for scene_index in range(num_scenes):
            scene_id = f"scene_{scene_index:03d}"
            writer.save_scene(
                scene_id,
                {"actors": {"target": np.zeros((1, 13), dtype=np.float32)}},
                {"seed": 1000 + scene_index, "env_id": "CANDIDBaseline-v1"},
            )
            for attempt in range(2):
                success = attempt == 0
                episode = _episode(seed=100 * scene_index + attempt, shift=0.01 * (scene_index + 1))
                key = f"traj_{episode_index}"
                group = handle.create_group(key)
                group.create_dataset("actions", data=episode["actions"])
                _write_obs(group.create_group("obs"), episode)
                flags = episode["trace_flags"]
                trace = _Trace(
                    {
                        "distractor_poses": episode["trace_poses"],
                        "distractor_linear_velocities": np.zeros(
                            (NUM_ACTIONS + 1, NUM_DISTRACTORS, 3), dtype=np.float32
                        ),
                        "distractor_angular_velocities": np.zeros(
                            (NUM_ACTIONS + 1, NUM_DISTRACTORS, 3), dtype=np.float32
                        ),
                        "object_sizes_m": np.full(NUM_DISTRACTORS, 0.04, dtype=np.float32),
                        "distractor_toppled": flags["distractor_toppled"],
                        "distractor_fallen": flags["distractor_fallen"],
                    },
                    [f"distractor_{i}" for i in range(NUM_DISTRACTORS)],
                )
                writer.save_trajectory(
                    f"{scene_id}_traj_{attempt:03d}",
                    scene_id,
                    trace,
                    {
                        "task_success": success,
                        "stage": "completed" if success else "check_grasp",
                        "saved_hdf5": True,
                        "hdf5_file": h5_name,
                        "hdf5_trajectory_key": key,
                        "control_mode": "pd_joint_pos",
                        "obs_mode": "state_dict",
                        "seed": 1000 + scene_index,
                    },
                )
                episode_index += 1
    validate_dataset(root)
    return root
