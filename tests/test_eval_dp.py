"""Safety criteria and aggregate rates do not depend on the GPU simulator."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candid.eval.metrics import SafetyLimits, evaluate_outcome, summarize_episodes


def trace(mean=0.05, max_value=0.08, topple=0, fall=0):
    return {"dr_mean": mean, "dr_max": max_value, "topple_count": topple, "fall_count": fall}


def test_safe_success_requires_task_success_and_low_disturbance_and_no_events():
    limits = SafetyLimits(0.1, 0.2)
    assert evaluate_outcome(True, trace(), limits)["safe_success"]
    assert not evaluate_outcome(False, trace(), limits)["safe_success"]
    assert not evaluate_outcome(True, trace(mean=0.101), limits)["safe_success"]
    assert not evaluate_outcome(True, trace(max_value=0.201), limits)["safe_success"]
    assert not evaluate_outcome(True, trace(topple=1), limits)["safe_success"]
    assert not evaluate_outcome(True, trace(fall=1), limits)["safe_success"]
    assert evaluate_outcome(False, trace(), limits)["safe"]


def test_inclusive_dr_limits_and_custom_range():
    lim = SafetyLimits(0.1, 0.25, dr_mean_min=0.03, dr_max_min=0.05)
    assert evaluate_outcome(True, trace(mean=0.03, max_value=0.25), lim)["safe"]
    assert not evaluate_outcome(True, trace(mean=0.029), lim)["safe"]
    assert not evaluate_outcome(True, trace(max_value=0.04), lim)["safe"]
    assert evaluate_outcome(True, trace(mean=0.03, max_value=0.25), SafetyLimits(0.1))["safe"]


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        SafetyLimits(-1)
    with pytest.raises(ValueError):
        SafetyLimits(math.nan)
    with pytest.raises(ValueError):
        SafetyLimits(0.1, 0.04, dr_max_min=0.05)
    with pytest.raises(ValueError):
        SafetyLimits(0.1, dr_mean_min=0.2)
    with pytest.raises(ValueError):
        SafetyLimits(0.1, dr_max_min=0.01)


def test_safe_success_vs_safety_and_success_aggregates():
    limits = SafetyLimits(0.1)
    rows = [
        {"scene_id": "a", "success_once": True, **trace(), **evaluate_outcome(True, trace(), limits)},
        {"scene_id": "b", "success_once": True, **trace(mean=0.3), **evaluate_outcome(True, trace(mean=0.3), limits)},
        {"scene_id": "c", "success_once": False, **trace(), **evaluate_outcome(False, trace(), limits)},
    ]
    result = summarize_episodes(rows, limits)
    assert result["success_rate"] == pytest.approx(2 / 3)
    assert result["safety_rate"] == pytest.approx(2 / 3)
    assert result["safe_success_rate"] == pytest.approx(1 / 3)
    assert result["success_once_rate"] == pytest.approx(2 / 3)
    assert result["scenes"] == 3
