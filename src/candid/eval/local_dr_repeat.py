"""Numerically tolerant, physically meaningful repeated-replay validation.

Keep the threshold strict, but never demand bit-for-bit equality of floating
point DR metrics. Event flags must remain exactly consistent. This module is
pure NumPy so it can be tested without ManiSkill or CUDA.
"""
from __future__ import annotations

import numpy as np


def compare_repeat(first, second, poses_first, poses_second, *,
                   pose_atol=1e-5, dr_atol=1e-4):
    if pose_atol <= 0 or dr_atol <= 0:
        raise ValueError("Replay tolerances must be positive")
    lhs = np.asarray(poses_first, dtype=np.float64)
    rhs = np.asarray(poses_second, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.size == 0:
        raise ValueError("Repeated replay pose arrays disagree in shape")
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise FloatingPointError("Non-finite repeated replay poses")

    max_pose = float(np.max(np.abs(lhs - rhs)))
    dr_errors = {}
    event_equal = True
    timeline_equal = first.get("executed_steps") == second.get("executed_steps")
    for h in (4, 8):
        name = f"h{h}"
        a, b = first.get(name), second.get(name)
        if a is None or b is None:
            timeline_equal &= a is None and b is None
            continue
        for metric in ("dr_mean", "dr_max"):
            x, y = float(a[metric]), float(b[metric])
            if not np.isfinite(x) or not np.isfinite(y):
                raise FloatingPointError("Non-finite repeated replay DR")
            dr_errors[f"{name}_{metric}"] = abs(x - y)
        for event in ("topple_count", "fall_count"):
            event_equal &= int(a[event]) == int(b[event])

    max_dr = max(dr_errors.values(), default=0.0)
    return {
        "passed": bool(
            timeline_equal and event_equal
            and max_pose <= pose_atol and max_dr <= dr_atol
        ),
        "pose_atol": pose_atol,
        "dr_atol": dr_atol,
        "max_pose_abs": max_pose,
        "max_dr_abs": max_dr,
        "dr_errors": dr_errors,
        "events_equal": bool(event_equal),
        "timeline_equal": bool(timeline_equal),
        "first_steps": first.get("executed_steps"),
        "repeat_steps": second.get("executed_steps"),
    }
