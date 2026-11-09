"""Regression checks for selected-scene evaluation seed stability."""
from __future__ import annotations

import ast
from pathlib import Path


def test_rollout_evaluation_keeps_original_scene_index_for_seed():
    # Importing rollout would bring optional simulation dependencies.
    source = (Path(__file__).resolve().parents[1] / "src/candid/eval/rollout.py").read_text()
    ast.parse(source)
    assert "original_test_index = {sid: index for index, sid in enumerate(full_test_ids)}" in source
    assert "seed = eval_seed + original_test_index[selected_id] * 100000 + repetition" in source


def test_guidance_diagnostic_source_is_syntactically_valid():
    root = Path(__file__).resolve().parents[1]
    for path in ("model/guidance/sampler.py", "model/guidance/timing.py",
                 "model/guidance/policy.py", "model/guidance/evaluate.py",
                 "scripts/analyze_guidance_diagnostics.py"):
        ast.parse((root / path).read_text())


def test_headless_state_dict_rollout_disables_renderer():
    source = (Path(__file__).resolve().parents[1] / "src/candid/eval/rollout.py").read_text()
    ast.parse(source)
    assert 'render_backend="gpu" if save_video else "none"' in source
    assert 'sim_backend="cpu"' in source
