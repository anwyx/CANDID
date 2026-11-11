"""Field-wise absolute replay error diagnostic for canonical observation vectors.

Canonical field ordering is inherited from candid.data.adapter. This module
never authorizes a replay mismatch; it only diagnoses rejected probes.
"""
from __future__ import annotations

import numpy as np

from candid.data.adapter import OBSERVATION_FIELDS


def observation_error_breakdown(replayed, archived):
    live = np.asarray(replayed, dtype=np.float64)
    source = np.asarray(archived, dtype=np.float64)
    if live.shape != source.shape or live.ndim != 2:
        raise ValueError("Observation windows must be matching [time, features]")
    # Scene1 canonical state_dict feature contract:
    # qpos and qvel: robot agent, tcp_pose, 16D target state,
    # 6 * 16D obstacle states, 6 mask values, and 2D goal.
    # Robot qpos/qvel can vary by agent; infer their combined size.
    lengths = {
        "tcp_pose": 7, "target_state": 16,
        "distractor_states": 6*16, "distractor_mask": 6,
        "goal_xy": 2,
    }
    fixed = sum(lengths.values())
    robot_total = live.shape[1] - fixed
    if robot_total < 2 or robot_total % 2:
        return {
            "largest_field": "unrecognized_observation_layout",
            "max_abs": float(np.max(np.abs(live-source))),
            "total_features": int(live.shape[1]),
        }
    lengths["robot_qpos"] = robot_total // 2
    lengths["robot_qvel"] = robot_total // 2
    offset = 0
    per_field = {}
    for name in OBSERVATION_FIELDS:
        width = lengths[name]
        diff = np.abs(live[:, offset:offset+width]-source[:, offset:offset+width])
        per_field[name] = {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "time_index_of_max": int(np.unravel_index(np.argmax(diff), diff.shape)[0]),
            "feature_index_of_max": int(np.unravel_index(np.argmax(diff), diff.shape)[1]),
        }
        offset += width
    if offset != live.shape[1]:
        raise RuntimeError("Observation feature length accounting error")
    biggest = max(per_field, key=lambda name: per_field[name]["max_abs"])
    return {
        "largest_field": biggest,
        "max_abs": per_field[biggest]["max_abs"],
        "per_field": per_field,
        "total_features": int(live.shape[1]),
    }
