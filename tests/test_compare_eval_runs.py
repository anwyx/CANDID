"""Paired policy comparison never mixes random suites or evaluation seeds."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_eval_runs import compare_runs


def make_run(path, sha="same", seed=2026, success=False, dr=.2):
    path.mkdir()
    summary = {
        "evaluation_source": "random_frozen_suite",
        "eval_suite_sha256": sha,
        "exec_horizon": 4,
        "max_steps": 300, "eval_seed": seed,
        "stop_on_success": True,
        "episodes_per_scene": 1,
        "safety_limits": {"dr_mean_min":0, "dr_mean_max":.2,
                          "dr_max_min":0, "dr_max_max":None},
    }
    (path / "summary.json").write_text(json.dumps(summary))
    episode = {
        "scene_id": "random_eval_900000_00000",
        "eval_seed": seed,
        "task_success": success,
        "safe": success,
        "safe_success": success,
        "dr_mean": dr, "dr_max":dr*2,
        "topple_count": 0, "fall_count": 0,
    }
    (path / "episodes.jsonl").write_text(json.dumps(episode)+"\n")
    return path


def test_paired_diff_and_identity(tmp_path):
    a = make_run(tmp_path/"a", success=False, dr=.2)
    b = make_run(tmp_path/"b", success=True, dr=.1)
    result = compare_runs(a,b,bootstrap_samples=50)
    assert result["metrics"]["task_success"]["delta_guided_minus_vanilla"] == 1
    assert abs(result["metrics"]["dr_mean"]["delta_guided_minus_vanilla"] + .1)<1e-9
    assert result["metrics"]["task_success"]["paired_bootstrap_95pct_ci"] == [1.,1.]


def test_reject_different_suite(tmp_path):
    a=make_run(tmp_path/"a")
    b=make_run(tmp_path/"b", sha="different")
    with pytest.raises(ValueError, match="eval_suite_sha256"):
        compare_runs(a,b)


def test_reject_different_seeds(tmp_path):
    a=make_run(tmp_path/"a")
    b=make_run(tmp_path/"b", seed=777)
    with pytest.raises(ValueError, match="eval_seed"):
        compare_runs(a,b)
