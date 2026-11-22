"""Scene-relative transfer waypoints for the experimental route-intent teacher.

These are TCP waypoint constraints, not whole-arm collision guarantees. Physics
still decides whether a planned and executed attempt succeeds or disturbs props.
"""
from __future__ import annotations

import numpy as np

INTENTS = ("direct", "left", "right", "over")


class RouteConstraintError(ValueError):
    """A requested route cannot fit within the configured waypoint bounds."""


def transfer_waypoints(start, goal, poses, dimensions, intent, *,
                       held_radius=0.025, clearance=0.012,
                       xy_limit=0.35, z_limit=0.42):
    """Return scene-relative TCP waypoints between the lifted and goal poses.

    The left/right corridor is relative to start -> goal, not world Y. The
    distractor footprint uses a conservative XY bounding circle. Only
    intermediate TCP poses are constrained; MPlib and the rollout determine
    reachability, arm-link contact, and final task success.
    """
    if intent not in INTENTS:
        raise ValueError(f"Unknown route intent: {intent}")
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    poses = np.asarray(poses, dtype=float)
    dims = np.asarray(dimensions, dtype=float)
    if (start.shape != (3,) or goal.shape != (3,) or poses.ndim != 2
            or poses.shape[1] < 3 or dims.shape != (len(poses), 3)
            or not all(np.isfinite(a).all() for a in (start, goal, poses, dims))
            or np.any(dims <= 0)):
        raise ValueError("Invalid transfer route geometry")
    if not (0 <= clearance <= .05 and 0 < held_radius <= .10
            and .20 <= xy_limit <= .60 and .20 <= z_limit <= .80):
        raise ValueError("Invalid route clearance or waypoint bounds")
    if intent == "direct":
        return np.empty((0, 3), dtype=float)

    delta = goal[:2] - start[:2]
    distance = float(np.linalg.norm(delta))
    if distance < .04:
        raise RouteConstraintError("Transfer is too short for a route corridor")
    forward = delta / distance
    left = np.array([-forward[1], forward[0]])
    base_z = max(start[2], goal[2])

    if intent == "over":
        # TCP carries the target below its fingers. Include its radius when
        # raising above the highest nearby distractor.
        nearby = []
        for pose, dim in zip(poses, dims):
            along = float(np.dot(pose[:2] - start[:2], forward))
            lateral = abs(float(np.dot(pose[:2] - start[:2], left)))
            if -.05 <= along <= distance + .05 and lateral <= .20:
                nearby.append(float(pose[2] + dim[2] / 2))
        height = max([base_z, *(top + held_radius + clearance for top in nearby)])
        points = np.array([.5 * (start + goal)], dtype=float)
        points[0, 2] = height
    else:
        # Include all objects intersecting the transfer strip. Offset the
        # middle corridor beyond their near-side footprint plus carried cube.
        edge = 0.0
        for pose, dim in zip(poses, dims):
            relative = pose[:2] - start[:2]
            along = float(np.dot(relative, forward))
            lateral = float(np.dot(relative, left))
            radius = float(np.linalg.norm(dim[:2]) / 2)
            if -.05 <= along <= distance + .05 and abs(lateral) <= .23:
                boundary = lateral + radius if intent == "left" else lateral - radius
                edge = max(edge, boundary) if intent == "left" else min(edge, boundary)
        offset = (max(.045, edge + held_radius + clearance) if intent == "left"
                  else min(-.045, edge - held_radius - clearance))
        points = np.array([start + (goal - start) * t for t in (.25, .75)])
        points[:, :2] += offset * left
        points[:, 2] = base_z

    if np.any(np.abs(points[:, :2]) > xy_limit) or np.any(points[:, 2] > z_limit):
        raise RouteConstraintError("Route waypoint exceeds configured XY/Z bounds")
    return points
