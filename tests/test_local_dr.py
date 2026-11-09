"""Lightweight static and NumPy checks for local DR probe contracts."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/candid/eval/local_dr.py"


def test_sources_parse():
    for path in (SOURCE, ROOT / "scripts/eval_local_dr.py"):
        ast.parse(path.read_text(encoding="utf-8"))


def test_local_dr_comparison_resets_same_simulator_snapshot():
    source = SOURCE.read_text(encoding="utf-8")
    assert "get_state_dict()" in source
    assert "freeze_state(state)" in source
    assert "trace.reset(" in source
    assert "disturbance_from_poses(" in source
    assert "start=0, stop=horizon" in source
    assert "torch.random.fork_rng" in source


def test_probe_step_validation_without_simulator():
    # Extract function source from AST so importing the full module is not
    # required on machines that do not have ManiSkill installed.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_steps")
    module = ast.Module(body=[fn], type_ignores=[])
    namespace = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    assert namespace["parse_steps"]("40,80,40") == [40, 80]
    with pytest.raises(ValueError):
        namespace["parse_steps"]("3,12")
    with pytest.raises(ValueError):
        namespace["parse_steps"]("-4")
