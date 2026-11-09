"""Pure-Python contracts for frozen random evaluation scene suites.

No SAPIEN or ManiSkill required: generation logic is exercised by a
single-environment smoke test on a simulation-enabled machine.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from candid.eval.scene_suite import load_suite, suite_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def _make_suite(tmp_path, n=2):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    rows = []
    for i in range(n):
        file = scenes / f"random_eval_123_{i:05d}.npz"
        np.savez_compressed(file, fake=np.asarray([i]))
        rows.append({
            "scene_id": f"random_eval_123_{i:05d}",
            "seed": 123 + i,
            "env_id": "TestEnv-v1",
            "env_module": "envs.example",
            "env_kwargs": {"randomize_layout": True},
            "generated_eval_suite": True,
            "initial_state": str(Path("scenes") / file.name),
        })
    (tmp_path / "scenes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    digest = suite_fingerprint(tmp_path)
    metadata = {
        "schema_version": 1, "source": "random_eval_suite",
        "num_scenes": n,
        "env_id": "TestEnv-v1", "env_module": "envs.example",
        "env_kwargs": {"randomize_layout": True}, "seed": 123,
        "max_episode_steps": 300, "suite_content_sha256": digest,
    }
    (tmp_path / "suite.json").write_text(json.dumps(metadata), encoding="utf-8")
    return rows


def test_suite_round_trip(tmp_path):
    rows = _make_suite(tmp_path)
    metadata, selected = load_suite(tmp_path)
    assert metadata["num_scenes"] == 2
    assert selected == rows
    assert metadata["suite_content_sha256"] == suite_fingerprint(tmp_path)


def test_snapshot_tampering_detected(tmp_path):
    _make_suite(tmp_path)
    np.savez_compressed(
        tmp_path / "scenes" / "random_eval_123_00000.npz",
        fake=np.asarray([9876]),
    )
    with pytest.raises(ValueError, match="changed"):
        load_suite(tmp_path)


def test_manifest_tampering_detected(tmp_path):
    _make_suite(tmp_path)
    path = tmp_path / "scenes.jsonl"
    contents = path.read_text()
    path.write_text(contents.replace('"seed": 123', '"seed": 999'))
    with pytest.raises(ValueError, match="changed"):
        load_suite(tmp_path)


def test_cannot_load_untagged_or_test_split_entries(tmp_path):
    rows = _make_suite(tmp_path)
    rows[0].pop("generated_eval_suite")
    p = tmp_path / "scenes.jsonl"
    p.write_text("".join(json.dumps(row) + "\n" for row in rows))
    meta = json.loads((tmp_path / "suite.json").read_text())
    meta["suite_content_sha256"] = suite_fingerprint(tmp_path)
    (tmp_path / "suite.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="not tagged"):
        load_suite(tmp_path)


def test_reject_unsafe_snapshot_path(tmp_path):
    _make_suite(tmp_path)
    path = tmp_path / "scenes.jsonl"
    s = path.read_text()
    path.write_text(s.replace("scenes/random_eval_123_00000.npz", "../escaped.npz"))
    with pytest.raises(ValueError, match="Unsafe"):
        suite_fingerprint(tmp_path)


def test_new_eval_interfaces_parse():
    for path in (
        "src/candid/eval/generate_suite.py",
        "src/candid/eval/scene_suite.py",
        "src/candid/eval/rollout.py",
        "scripts/generate_eval_scenes.py",
        "model/vanilla_dp/evaluate.py",
        "model/guidance/evaluate.py",
        "model/integration/evaluate.py",
    ):
        ast.parse((ROOT / path).read_text(encoding="utf-8"))


def test_all_policies_look_up_frozen_suite():
    rollout = (ROOT / "src/candid/eval/rollout.py").read_text()
    assert "load_suite(snapshot_root)" in rollout
    assert "suite_meta[\"suite_content_sha256\"]" in rollout
    assert "random_frozen_suite" in rollout
    assert "original_test_index = {sid: index for index, sid in enumerate(full_test_ids)}" in rollout
    for path in (
        "model/vanilla_dp/evaluate.py",
        "model/guidance/evaluate.py",
        "model/integration/evaluate.py",
    ):
        source = (ROOT / path).read_text()
        assert "--eval-suite" in source
        assert "eval_suite=args.eval_suite" in source
