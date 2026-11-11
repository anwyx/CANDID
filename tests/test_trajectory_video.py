"""Offline GRITS-style trace rendering, with no simulator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candid.vis.trajectory_video import load_trajectories, render_frames, render_trajectory_video


def _episode(x_shift, steps=8):
    time = np.linspace(0, 1, steps, dtype=np.float32)
    tcp = np.zeros((steps, 7), dtype=np.float32)
    tcp[:, 0] = -0.12 + 0.18 * time
    tcp[:, 1] = x_shift
    tcp[:, 2] = 0.12
    tcp[:, 3] = 1
    target = tcp.copy()
    target[:, 2] = 0.02
    goal = np.zeros((steps, 3), dtype=np.float32)
    goal[:, 0] = 0.1
    poses = np.zeros((steps, 1, 7), dtype=np.float32)
    poses[:, 0, 0] = -0.05
    poses[:, 0, 1] = 0.08 + 0.04 * time * (1 if x_shift >= 0 else 0)
    poses[:, 0, 2] = 0.02
    poses[:, 0, 3] = 1
    toppled = np.zeros((steps, 1), dtype=bool)
    toppled[-1, 0] = x_shift < 0
    return {
        "tcp_pose": tcp,
        "target_pose": target,
        "goal_pos": goal,
        "distractor_poses": poses,
        "object_sizes_m": np.array([0.04], dtype=np.float32),
        "distractor_toppled": toppled,
        "distractor_fallen": np.zeros((steps, 1), dtype=bool),
    }


def _save(path, episode):
    np.savez_compressed(path, **episode)


def test_single_archive_is_one_episode(tmp_path):
    path = tmp_path / "trajectory.npz"
    _save(path, _episode(0.0))
    loaded = load_trajectories(path)
    assert len(loaded) == 1
    assert loaded[0]["tcp_pose"].shape == (8, 7)


def test_stacked_archive_is_two_episodes(tmp_path):
    path = tmp_path / "trajectory.npz"
    first, second = _episode(0.04), _episode(-0.04)
    stacked = {
        key: np.stack((first[key], second[key]), axis=0)
        for key in first
    }
    np.savez_compressed(path, **stacked)
    loaded = load_trajectories(path)
    assert len(loaded) == 2
    assert loaded[1]["tcp_pose"][0, 1] == np.float32(-0.04)


def test_two_traces_are_both_visible(tmp_path):
    frames = render_frames([_episode(0.05), _episode(-0.05)], width=320, height=320)
    image = frames[len(frames) // 2]
    blue = np.logical_and(image[:, :, 2] > 180, image[:, :, 0] < 120)
    red = np.logical_and(image[:, :, 0] > 180, image[:, :, 2] < 120)
    assert blue.sum() > 20
    assert red.sum() > 20
    summary = render_trajectory_video(
        _write_pair(tmp_path),
        tmp_path / "out",
        width=160,
        height=160,
        fps=5,
    )
    assert Path(summary["video"]).stat().st_size > 1000
    assert Path(summary["gif"]).stat().st_size > 1000
    assert summary["episodes"][0]["topple_count"] == 0
    assert summary["episodes"][1]["topple_count"] == 1


def _write_pair(tmp_path):
    path = tmp_path / "pair.npz"
    first, second = _episode(0.04), _episode(-0.04)
    np.savez_compressed(path, **{key: np.stack((first[key], second[key]), axis=0) for key in first})
    return path
