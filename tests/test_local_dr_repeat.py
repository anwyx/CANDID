"""Numerical regression checks for baseline replay reproducibility."""
from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candid.eval.local_dr_repeat import compare_repeat


def _metrics(value=0.1, topple=0):
    return {
        "executed_steps": 8,
        "h4": {"dr_mean": value, "dr_max": value * 2,
               "topple_count": topple, "fall_count": 0},
        "h8": {"dr_mean": value * 2, "dr_max": value * 4,
               "topple_count": topple, "fall_count": 0},
    }


def test_repeated_replay_allows_tiny_fp_rounding_errors():
    poses = np.ones((9, 2, 7))
    a, b = _metrics(), _metrics(value=0.10000001)
    result = compare_repeat(a, b, poses, poses + 1e-7)
    assert result["passed"]
    assert result["max_dr_abs"] > 0


def test_detects_actual_physics_divergence():
    poses = np.ones((9, 2, 7))
    assert not compare_repeat(
        _metrics(), _metrics(value=0.101),
        poses, poses + 1e-3,
    )["passed"]


def test_event_disagreement_is_never_tolerated():
    poses = np.ones((9, 2, 7))
    result = compare_repeat(_metrics(), _metrics(topple=1), poses, poses)
    assert not result["passed"]
    assert not result["events_equal"]


def test_timeline_disagreement_is_never_tolerated():
    poses = np.ones((9, 2, 7))
    a, b = _metrics(), _metrics()
    b["executed_steps"] = 7
    result = compare_repeat(a, b, poses, poses)
    assert not result["passed"]
    assert not result["timeline_equal"]


def test_quaternion_sign_flip_is_same_rotation():
    poses = np.zeros((9, 2, 7))
    poses[..., 3] = 1.0
    flipped = poses.copy()
    flipped[..., 3:] *= -1
    result = compare_repeat(_metrics(), _metrics(), poses, flipped)
    assert result["passed"]
    assert result["max_rotation_rad"] == 0.0


def test_small_quaternion_component_difference_is_under_angle_tolerance():
    poses = np.zeros((9, 2, 7))
    poses[..., 3] = 1.0
    shifted = poses.copy()
    shifted[..., 4] = 1.3275938e-5
    result = compare_repeat(_metrics(), _metrics(), poses, shifted)
    assert result["passed"]
    assert result["max_rotation_rad"] < 5e-4


def test_large_orientation_change_fails():
    poses = np.zeros((9, 2, 7))
    poses[..., 3] = 1.0
    shifted = poses.copy()
    shifted[..., 4] = 0.01
    result = compare_repeat(_metrics(), _metrics(), poses, shifted)
    assert not result["passed"]


def test_substantive_dr_difference_fails():
    poses = np.zeros((9, 2, 7))
    poses[..., 3] = 1.0
    result = compare_repeat(_metrics(), _metrics(value=0.101), poses, poses)
    assert not result["passed"]
