"""Static regression checks for random evaluation integrity guards."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_random_suite_uses_current_spec_and_rejects_initial_success():
    source = (ROOT / "src/candid/eval/rollout.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "current_spec.to_dict()" in source
    assert "actual_spec != saved_spec" in source
    assert 'env.unwrapped.evaluate()["success"]' in source
    assert "begins in " in source
    assert '"stable_initial_goal_pos"' in source


def test_offline_audit_tool_is_valid_python():
    source = (ROOT / "scripts/audit_eval_suite.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "--old-dataset" in source
    assert "success_but_outside_goal_xy" in source
    assert "old_scene_spec_equals_new" in source
