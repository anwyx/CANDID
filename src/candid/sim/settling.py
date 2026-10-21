"""Pose-based settling of dynamic objects before a CANDID episode.

The data collector and disturbance monitor are not involved during this phase.
The caller supplies a physics-step function and a pose snapshot function.
Snapshots are [N,7] arrays in world frame with wxyz quaternions.

Two checks are required:
  (1) bounded PEAK displacement/rotation during a recent sliding window
  (2) bounded PEAK displacement/rotation during a subsequent idle verification
A brief velocity spike is not automatically a failure when the object barely
moves; persistent sliding or rolling fails the pose checks.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class SettlingConfig:
    max_steps: int = 180
    min_steps: int = 20
    window_steps: int = 15
    window_translation_m: float = 0.0007
    window_rotation_deg: float = 1.0
    verify_steps: int = 30
    verify_translation_m: float = 0.0010
    verify_rotation_deg: float = 1.5

    def __post_init__(self):
        if not (1 <= self.window_steps < self.max_steps):
            raise ValueError("window_steps must be between 1 and max_steps-1")
        if self.min_steps < 0 or self.verify_steps < 1 or self.max_steps < self.min_steps + self.window_steps + self.verify_steps:
            raise ValueError("max_steps must allow min_steps + window_steps + verify_steps")
        if min(self.window_translation_m, self.window_rotation_deg,
               self.verify_translation_m, self.verify_rotation_deg) <= 0:
            raise ValueError("Pose stability thresholds must be positive")


def _validate_pose(pose):
    arr = np.asarray(pose, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 7 or not np.isfinite(arr).all():
        raise ValueError(f"Expected finite pose array [N,7], got {arr.shape}")
    return arr.copy()


def peak_motion(reference, poses):
    """Max absolute translation [m] and quaternion angle [degrees] over time.

    Returns (max_translation, max_angle_degrees, worst_translation_actor,
    worst_rotation_actor). Quaternion double-cover handled by abs(dot).
    """
    reference = _validate_pose(reference)
    frames = np.stack([_validate_pose(p) for p in poses], axis=0)
    if frames.shape[1:] != reference.shape:
        raise ValueError("All pose frames must have the same object order/shape")
    movement = np.linalg.norm(frames[:, :, :3] - reference[None, :, :3], axis=-1)
    q0 = reference[:, 3:] / np.maximum(np.linalg.norm(reference[:, 3:], axis=-1, keepdims=True), 1e-12)
    q1 = frames[:, :, 3:] / np.maximum(np.linalg.norm(frames[:, :, 3:], axis=-1, keepdims=True), 1e-12)
    dots = np.sum(q1 * q0[None, :, :], axis=-1)
    angles_deg = np.rad2deg(2 * np.arccos(np.clip(np.abs(dots), 0, 1)))
    tmax = movement.max(axis=0)
    rmax = angles_deg.max(axis=0)
    return float(tmax.max()), float(rmax.max()), int(tmax.argmax()), int(rmax.argmax())


def settle_scene(step: Callable[[], None], snapshot: Callable[[], np.ndarray],
                 config: SettlingConfig, object_names=None):
    """Run unrecorded physics steps until both pose-based tests pass.

    Must be called INSIDE env.reset AFTER super().reset, before external wrappers
    observe the episode start. Single-environment CPU collector only.
    """
    initial = _validate_pose(snapshot())
    n = len(initial)
    names = list(object_names) if object_names is not None else [str(i) for i in range(n)]
    if len(names) != n:
        raise ValueError("Object-name count does not match pose count")
    history = deque([initial], maxlen=config.window_steps + 1)
    elapsed = 0
    candidates = 0
    best_translation = float("inf")
    best_rotation = float("inf")
    last_result = None
    while elapsed < config.max_steps:
        step()
        elapsed += 1
        current = _validate_pose(snapshot())
        history.append(current)
        if elapsed < config.min_steps or len(history) < config.window_steps + 1:
            continue
        wtrans, wrot, wit, wir = peak_motion(history[0], list(history)[1:])
        best_translation = min(best_translation, wtrans)
        best_rotation = min(best_rotation, wrot)
        last_result = (wtrans, wrot, wit, wir)
        if wtrans > config.window_translation_m or wrot > config.window_rotation_deg:
            continue
        if elapsed + config.verify_steps > config.max_steps:
            break
        candidates += 1
        verify_start = current
        verify_poses = []
        for _ in range(config.verify_steps):
            step()
            elapsed += 1
            current = _validate_pose(snapshot())
            verify_poses.append(current)
        vtrans, vrot, vit, vir = peak_motion(verify_start, verify_poses)
        if vtrans <= config.verify_translation_m and vrot <= config.verify_rotation_deg:
            return {
                "idle_stable": True,
                "settle_steps": elapsed,
                "window_steps": config.window_steps,
                "window_peak_translation_m": wtrans,
                "window_peak_rotation_deg": wrot,
                "verify_steps": config.verify_steps,
                "idle_translation_m": vtrans,
                "idle_rotation_deg": vrot,
                "settling_candidate_count": candidates,
            }
        # Failed verification: the scene is still moving. Keep waiting.
        last_result = (vtrans, vrot, vit, vir)
        history.clear()
        history.append(current)

    if last_result is None:
        detail = "no complete sliding window"
    else:
        trans, rot, it, ir = last_result
        detail = (f"last peak translation={trans * 1000:.3f} mm ({names[it]}), "
                  f"last peak rotation={rot:.3f} deg ({names[ir]})")
    raise RuntimeError(
        f"Scene still moving after {elapsed}/{config.max_steps} unrecorded control steps; "
        f"{detail}; candidates verified={candidates}; "
        f"best window translation={best_translation*1000:.3f} mm, "
        f"best window rotation={best_rotation:.3f} deg. "
        "Reject this scene or inspect the actor contacts/layout; do not record a disturbance label."
    )
