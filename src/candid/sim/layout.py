"""Seeded rejection sampler based on conservative actor footprint circles."""
from __future__ import annotations
import numpy as np


def sample_layout(spec, radii, rng, randomize=True):
    """radii = [target_radius, *distractor_radii] in metres.

    Returns layout with stable identity/order, independent of placement order.
    The goal disk is VISUAL ONLY: reserve a physical landing footprint rather
    than excluding its decorative marker radius.
    """
    radii = np.asarray(radii, dtype=float)
    if radii.shape != (1 + len(spec.distractors),) or np.any(radii <= 0):
        raise ValueError("One positive footprint radius per target/distractor is required")
    if not randomize:
        return {
            "target_xy": list(spec.target.fixed_xy),
            "target_yaw": spec.target.fixed_yaw,
            "goal_xy": list(spec.goal_fixed_xy),
            "distractors_xy": [list(o.fixed_xy) for o in spec.distractors],
            "distractors_yaw": [o.fixed_yaw for o in spec.distractors],
        }
    n = len(spec.distractors)
    for _ in range(spec.max_layout_attempts):
        target_xy = np.asarray(spec.target.region.sample(rng))
        goal_xy = np.asarray(spec.goal_region.sample(rng))
        placed = []
        result_xy = [None] * n
        result_yaw = [None] * n
        order = sorted(range(n), key=lambda i: -radii[i+1])
        failed = False
        for i in order:
            obj = spec.distractors[i]
            rad = radii[i+1]
            for _ in range(spec.max_position_samples):
                pos = np.asarray(obj.region.sample(rng))
                if np.linalg.norm(pos - target_xy) < rad+radii[0]+spec.min_surface_gap:
                    continue
                if np.linalg.norm(pos - goal_xy) < rad+radii[0]+spec.goal_landing_gap:
                    continue
                if any(np.linalg.norm(pos-prev) < rad+other_rad+spec.min_surface_gap for prev,other_rad in placed):
                    continue
                placed.append((pos, rad))
                result_xy[i] = pos.tolist()
                result_yaw[i] = float(rng.uniform(-np.pi, np.pi))
                break
            else:
                failed = True
                break
        if not failed:
            return {
                "target_xy": target_xy.tolist(),
                "target_yaw": float(rng.uniform(-np.pi, np.pi)),
                "goal_xy": goal_xy.tolist(),
                "distractors_xy": result_xy,
                "distractors_yaw": result_yaw,
            }
    raise RuntimeError(
        f"Could not sample a collision-free {spec.name} layout. "
        "Increase XY regions or reduce object count/gaps."
    )
