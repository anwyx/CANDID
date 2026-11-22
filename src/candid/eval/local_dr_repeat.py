"""Check repeated fixed-state physical replay in meaningful units.

Tiny floating-point differences are allowed; substantive physical divergence,
changed event counts, or changed episode lengths still fail the check.
"""
from __future__ import annotations

import numpy as np


def compare_repeat(first, second, poses_first, poses_second, *,
                   translation_atol=5e-5, rotation_atol=5e-4,
                   dr_atol=1e-6):
    if min(translation_atol, rotation_atol, dr_atol) <= 0:
        raise ValueError("Replay tolerances must be positive")
    a = np.asarray(poses_first, dtype=np.float64)
    b = np.asarray(poses_second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 7 or not a.size:
        raise ValueError("Replay poses need matching nonempty [T+1,N,7] shapes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("Replay poses contain NaN or Inf")

    max_translation = float(np.linalg.norm(a[..., :3] - b[..., :3], axis=-1).max())
    qa, qb = a[..., 3:], b[..., 3:]
    norms_a = np.linalg.norm(qa, axis=-1, keepdims=True)
    norms_b = np.linalg.norm(qb, axis=-1, keepdims=True)
    if (norms_a <= 1e-12).any() or (norms_b <= 1e-12).any():
        raise ValueError("Zero quaternion in replay poses")
    qa, qb = qa / norms_a, qb / norms_b
    dot = np.sum(qa * qb, axis=-1)
    max_rotation = float(
        (2 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))).max()
    )

    dr_errors = {}
    events_equal = True
    timeline_equal = first.get("executed_steps") == second.get("executed_steps")
    for h in (4, 8):
        key = f"h{h}"
        p, q = first.get(key), second.get(key)
        if p is None or q is None:
            timeline_equal &= (p is None and q is None)
            continue
        for metric in ("dr_mean", "dr_max"):
            x, y = float(p[metric]), float(q[metric])
            if not np.isfinite(x) or not np.isfinite(y):
                raise FloatingPointError("Replay DR contains NaN or Inf")
            dr_errors[f"{key}_{metric}"] = abs(x - y)
        for event in ("topple_count", "fall_count"):
            events_equal &= int(p[event]) == int(q[event])

    max_dr = max(dr_errors.values(), default=0.0)
    return {
        "passed": bool(
            timeline_equal and events_equal
            and max_translation <= translation_atol
            and max_rotation <= rotation_atol
            and max_dr <= dr_atol
        ),
        "translation_atol_m": float(translation_atol),
        "rotation_atol_rad": float(rotation_atol),
        "dr_atol": float(dr_atol),
        "max_pose_abs": float(np.max(np.abs(a - b))),
        "max_translation_m": max_translation,
        "max_rotation_rad": max_rotation,
        "max_dr_abs": max_dr,
        "dr_errors": dr_errors,
        "events_equal": bool(events_equal),
        "timeline_equal": bool(timeline_equal),
        "first_steps": first.get("executed_steps"),
        "repeat_steps": second.get("executed_steps"),
    }
