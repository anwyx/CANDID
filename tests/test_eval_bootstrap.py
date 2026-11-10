"""Protect random-suite bootstrap bypass without weakening explicit settling."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_constructor_only_bootstrap_guard():
    file = (ROOT / "src/candid/sim/environment.py").read_text(encoding="utf-8")
    ast.parse(file)
    assert "skip_constructor_settling=False" in file
    assert "self._in_constructor = True" in file
    assert "self._in_constructor = False" in file
    assert "if self._in_constructor and self._skip_constructor_settling:" in file
    assert "self._initial_stability = settle_scene(" in file
    assert "Invalid initial scene after settling:" in file


def test_suite_generation_and_replay_use_bootstrap_only_bypass():
    for file in ("src/candid/eval/generate_suite.py", "src/candid/eval/rollout.py"):
        code = (ROOT / file).read_text(encoding="utf-8")
        ast.parse(code)
        assert "skip_constructor_settling" in code
    gen = (ROOT / "src/candid/eval/generate_suite.py").read_text()
    assert "trace.reset(seed=scene_seed)" in gen
    replay = (ROOT / "src/candid/eval/rollout.py").read_text()
    assert 'options={"reset_to_env_states": {"env_states": saved}}' in replay
    assert "_check_initial_scene(scene, env.unwrapped, trace)" in replay
