"""Multi-environment suite merging and backward compatibility checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from candid.eval.scene_suite import load_suite, suite_fingerprint
from scripts.merge_eval_suites import merge_suites
from scripts.summarize_eval_groups import summarize_groups


def make_source(root, label, env_id, count=2):
    (root / "scenes").mkdir(parents=True)
    spec = {"name": label, "distractors": [], "goal_xy_tolerance": 0.045}
    with (root / "scenes.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(count):
            state = f"scenes/scene{i}.npz"
            np.savez_compressed(root / state, qpos=np.array([i], dtype=np.float32))
            row = {
                "scene_id": f"random_eval_{i}",
                "seed": 900000 + i,
                "env_id": env_id,
                "env_module": f"envs.{label}",
                "env_kwargs": {"randomize_layout": True},
                "generated_eval_suite": True,
                "scene_spec": spec,
                "initial_state": state,
                "distractor_ids": ["one"],
            }
            handle.write(json.dumps(row)+"\n")
    manifest = {
        "schema_version": 1, "source": "random_eval_suite",
        "env_id": env_id, "env_module": f"envs.{label}",
        "env_kwargs": {"randomize_layout": True},
        "num_scenes": count, "seed": 900000, "max_episode_steps": 300,
        "suite_content_sha256": suite_fingerprint(root),
    }
    (root / "suite.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_merge_two_sources_different_envs(tmp_path):
    one = make_source(tmp_path / "one", "one", "TestOne-v1")
    two = make_source(tmp_path / "two", "two", "TestTwo-v1")
    output = tmp_path / "out"
    merged = merge_suites([("scene1", one), ("scene2", two)], output, expected_scenes=2)
    assert merged["num_scenes"] == 4
    assert merged["source"] == "multi_env_eval_suite"
    meta, rows = load_suite(output)
    assert len(rows) == 4
    assert len({x["scene_id"] for x in rows}) == 4
    assert {x["suite_group"] for x in rows} == {"scene1", "scene2"}
    for row in rows:
        assert (output / row["initial_state"]).exists()
    assert meta["suite_content_sha256"] == suite_fingerprint(output)


def test_original_suite_still_loads(tmp_path):
    source = make_source(tmp_path / "source", "one", "TestOne-v1")
    meta, rows = load_suite(source)
    assert meta["source"] == "random_eval_suite"
    assert len(rows) == 2


def test_conflicting_env_id_rejected(tmp_path):
    one = make_source(tmp_path / "one", "one", "SameID-v1")
    two = make_source(tmp_path / "two", "two", "SameID-v1")
    with pytest.raises(ValueError, match="Conflicting registrations"):
        merge_suites([("scene1", one), ("scene2", two)], tmp_path / "out", expected_scenes=2)


def test_missing_input_cannot_create_partial_output(tmp_path):
    one = make_source(tmp_path / "one", "one", "TestOne-v1")
    two = make_source(tmp_path / "two", "two", "TestTwo-v1")
    (two / "scenes" / "scene1.npz").unlink()
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        merge_suites([("scene1", one), ("scene2", two)], out, expected_scenes=2)
    assert not out.exists()


def test_bad_per_source_count_rejected(tmp_path):
    one = make_source(tmp_path / "one", "one", "TestOne-v1")
    two = make_source(tmp_path / "two", "two", "TestTwo-v1")
    with pytest.raises(ValueError, match="expected 50"):
        merge_suites([("scene1", one), ("scene2", two)], tmp_path / "out")


def test_grouped_summary(tmp_path):
    one = make_source(tmp_path / "one", "one", "TestOne-v1")
    two = make_source(tmp_path / "two", "two", "TestTwo-v1")
    combined = tmp_path / "out"
    meta = merge_suites([("scene1", one), ("scene2", two)], combined, expected_scenes=2)
    _, rows = load_suite(combined)
    resultdir = tmp_path / "results"
    resultdir.mkdir()
    (resultdir / "summary.json").write_text(json.dumps({
        "eval_suite_sha256": meta["suite_content_sha256"],
        "episodes_per_scene": 1,
        "safety_limits": {"dr_mean_min": 0., "dr_mean_max": 0.2,
                          "dr_max_min": 0., "dr_max_max": None}
    }), encoding="utf-8")
    with (resultdir / "episodes.jsonl").open("w") as handle:
        for row in rows:
            safe = row["suite_group"] == "scene1"
            result = {
                "scene_id": row["scene_id"], "task_success": True,
                "success_once": True, "safe": safe, "safe_success": safe,
                "dr_mean": 0.01 if safe else 0.3, "dr_max": 0.02 if safe else 0.4,
                "topple_count": 0 if safe else 1, "fall_count": 0,
            }
            handle.write(json.dumps(result)+"\n")
    summary = summarize_groups(combined, resultdir)
    assert summary["scene1"]["safety_rate"] == 1.
    assert summary["scene2"]["safety_rate"] == 0.
