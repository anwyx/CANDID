"""Unambiguous safety and success definitions for CANDID closed-loop evaluation."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyLimits:
    """Acceptable DR interval is [0, dr_mean_max], optionally [0, dr_max_max]."""
    dr_mean_max: float
    dr_max_max: float | None = None
    dr_mean_min: float = 0.0
    dr_max_min: float = 0.0

    def __post_init__(self):
        for name, value in (("dr_mean_min", self.dr_mean_min), ("dr_mean_max", self.dr_mean_max),
                            ("dr_max_min", self.dr_max_min), ("dr_max_max", self.dr_max_max)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite nonnegative value")
        if self.dr_mean_min > self.dr_mean_max:
            raise ValueError("DR Mean lower bound exceeds upper bound")
        if self.dr_max_max is not None and self.dr_max_min > self.dr_max_max:
            raise ValueError("DR Max lower bound exceeds upper bound")
        if self.dr_max_max is None and self.dr_max_min != 0:
            raise ValueError("Set a DR Max upper bound when using a nonzero lower bound")

    def as_dict(self):
        return {"dr_mean_min": self.dr_mean_min, "dr_mean_max": self.dr_mean_max,
                "dr_max_min": self.dr_max_min, "dr_max_max": self.dr_max_max}


def evaluate_outcome(task_success: bool, disturbance: dict, limits: SafetyLimits) -> dict:
    """Report safety independently from task success; both are needed for safe success."""
    mean = float(disturbance["dr_mean"])
    maximum = float(disturbance["dr_max"])
    if not math.isfinite(mean) or not math.isfinite(maximum):
        raise ValueError("Non-finite disturbance metrics")
    topple = int(disturbance.get("topple_count", 0))
    fall = int(disturbance.get("fall_count", 0))
    if topple < 0 or fall < 0:
        raise ValueError("Invalid negative event count")
    within_mean = limits.dr_mean_min <= mean <= limits.dr_mean_max
    within_max = limits.dr_max_max is None or limits.dr_max_min <= maximum <= limits.dr_max_max
    no_events = topple == 0 and fall == 0
    safe = within_mean and within_max and no_events
    return {
        "task_success": bool(task_success),
        "no_topple_or_fall": no_events,
        "dr_mean_within_limit": within_mean,
        "dr_max_within_limit": within_max,
        "safe": safe,
        "safe_success": bool(task_success) and safe,
    }


def summarize_episodes(rows: list[dict], limits: SafetyLimits) -> dict:
    if not rows:
        raise ValueError("No evaluation episodes")
    n = len(rows)
    return {
        "episodes": n,
        "scenes": len({row["scene_id"] for row in rows}),
        "success_rate": sum(bool(row["task_success"]) for row in rows) / n,
        "success_once_rate": sum(bool(row["success_once"]) for row in rows) / n,
        "safety_rate": sum(bool(row["safe"]) for row in rows) / n,
        "safe_success_rate": sum(bool(row["safe_success"]) for row in rows) / n,
        "mean_dr_mean": sum(float(row["dr_mean"]) for row in rows) / n,
        "mean_dr_max": sum(float(row["dr_max"]) for row in rows) / n,
        "total_topple": sum(int(row["topple_count"]) for row in rows),
        "total_fall": sum(int(row["fall_count"]) for row in rows),
        "safety_limits": limits.as_dict(),
    }
