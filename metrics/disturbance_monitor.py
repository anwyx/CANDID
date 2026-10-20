"""Non-intrusive, simulator-truth disturbance recorder for CANDID.

The wrapper never modifies observations or actions. The ground-truth poses
are used ONLY for logging/training targets and evaluation, not policy inputs.

A custom environment may expose these OPTIONAL methods on ``env.unwrapped``:

    get_distractor_poses_for_logging() -> [1, N, 7] or [N, 7]
        Each pose is [x, y, z, qw, qx, qy, qz] in world coordinates.

    get_distractor_sizes_for_logging() -> [N]
        Each positive size is a characteristic LENGTH IN METRES. For cubes,
        use side length; for other geometries, document the chosen convention.

If the first method is missing, metric/trace recording is disabled but the
runner and the motion planner can still execute. If sizes are missing and
DISTRACTOR_HALF_SIZE is unavailable, raw motion is retained while normalized
continuous DR is ``None``. Missing measurements are never labelled as zero.

IMPORTANT: Supports ONE simulated environment at a time (CPU + MPlib runner).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


def continuous_disturbance_score(
    translation_m: np.ndarray,
    rotation_rad: np.ndarray,
    object_sizes_m: np.ndarray,
    translation_weight: float = 0.7,
    rotation_weight: float = 0.3,
) -> dict[str, Any]:
    """Calculate continuous mean/worst disturbance from per-object motion.

    This pure function can later be called by chunk-level preprocessing too.
    Both inputs must refer to the SAME time interval/reference frame. The
    weights are metric conventions, not fitted physical parameters.
    """
    trans = np.asarray(translation_m, dtype=np.float64).reshape(-1)
    rot = np.asarray(rotation_rad, dtype=np.float64).reshape(-1)
    sizes = np.asarray(object_sizes_m, dtype=np.float64).reshape(-1)
    if not (len(trans) == len(rot) == len(sizes)):
        raise ValueError("translation, rotation and sizes must have equal lengths")
    if np.any(~np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError("Every object characteristic size must be positive and finite")
    if translation_weight < 0 or rotation_weight < 0:
        raise ValueError("Disturbance weights must be nonnegative")

    per_object = translation_weight * trans / sizes + rotation_weight * rot / np.pi
    return {
        "continuous_dr": float(per_object.mean()) if len(per_object) else 0.0,
        "d_max": float(per_object.max()) if len(per_object) else 0.0,
        "per_object_disturbance": per_object.tolist(),
    }


class DisturbanceMonitor(gym.Wrapper):
    """Record per-step distractor poses and derive episode-level consequence.

    ``peak_translation`` means the largest distance from *episode-initial*
    pose, NOT path length; independently measured translation/rotation peaks
    can occur at different timestamps.
    """

    def __init__(
        self,
        env: gym.Env,
        translation_weight: float = 0.7,
        rotation_weight: float = 0.3,
        table_height_m: float = 0.0,
        topple_angle_deg: float = 60.0,
        fall_below_table_m: float = 0.03,
    ):
        super().__init__(env)
        self.translation_weight = float(translation_weight)
        self.rotation_weight = float(rotation_weight)
        self.table_height_m = float(table_height_m)
        self.topple_angle_deg = float(topple_angle_deg)
        self.fall_below_table_m = float(fall_below_table_m)
        if self.translation_weight < 0 or self.rotation_weight < 0:
            raise ValueError("Disturbance weights cannot be negative")
        if not 0 <= self.topple_angle_deg <= 180:
            raise ValueError("topple_angle_deg must be between 0 and 180")
        self.reset_count = 0
        self.steps = 0
        self.initial_pose: np.ndarray | None = None
        self.latest_pose: np.ndarray | None = None
        self.object_sizes_m: np.ndarray | None = None
        self.peak_translation: np.ndarray | None = None
        self.peak_rotation: np.ndarray | None = None
        self.ever_toppled: np.ndarray | None = None
        self.ever_fell: np.ndarray | None = None
        self.pose_trace: list[np.ndarray] = []
        self.available = False

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        """Accept NumPy arrays or tensors from ManiSkill."""
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    def _poses(self) -> np.ndarray | None:
        """Fetch current simulator truth, never add it to observations."""
        base = self.env.unwrapped
        getter = getattr(base, "get_distractor_poses_for_logging", None)
        if callable(getter):
            poses = self._as_numpy(getter())
        elif hasattr(base, "clutter_cubes"):
            # Backward-compatible fallback for teammate environments cloned
            # from the baseline that expose the same actor list.
            poses = np.stack(
                [self._as_numpy(obj.pose.raw_pose).reshape(-1, 7)[0]
                 for obj in base.clutter_cubes],
                axis=0,
            ) if base.clutter_cubes else np.zeros((0, 7), dtype=np.float64)
        else:
            return None

        if poses.ndim == 3:
            if poses.shape[0] != 1:
                raise ValueError("DisturbanceMonitor only supports num_envs=1")
            poses = poses[0]
        if poses.ndim != 2 or poses.shape[-1] != 7:
            raise ValueError(f"Expected distractor pose array [N,7], got {poses.shape}")
        return poses.copy()

    def _sizes(self, n: int) -> np.ndarray | None:
        """Find geometry normalization scales; return None if not available."""
        base = self.env.unwrapped
        getter = getattr(base, "get_distractor_sizes_for_logging", None)
        if callable(getter):
            scales = self._as_numpy(getter()).reshape(-1)
        elif hasattr(base, "DISTRACTOR_HALF_SIZE"):
            # Baseline has equally sized cubes; the characteristic scale is
            # their full side length, exactly as in the previous script.
            scales = np.full(n, 2.0 * float(base.DISTRACTOR_HALF_SIZE))
        else:
            return None
        if len(scales) != n or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError(f"Expected {n} positive characteristic sizes in metres, got {scales}")
        return scales

    @staticmethod
    def _rotation_angle(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """Smallest SO(3) angular separation, quaternions in wxyz order."""
        q0 = q0 / np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-9)
        q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-9)
        half_cos = np.abs(np.sum(q0 * q1, axis=-1))
        return 2.0 * np.arccos(np.clip(half_cos, 0.0, 1.0))

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.reset_count += 1
        self.steps = 0
        poses = self._poses()
        self.initial_pose = poses
        self.latest_pose = poses.copy() if poses is not None else None
        self.available = poses is not None
        self.pose_trace = [] if poses is None else [poses.copy()]
        self.object_sizes_m = None
        self.peak_translation = self.peak_rotation = None
        self.ever_toppled = self.ever_fell = None
        if self.available:
            n = len(poses)
            self.object_sizes_m = self._sizes(n)
            self.peak_translation = np.zeros(n, dtype=np.float64)
            self.peak_rotation = np.zeros(n, dtype=np.float64)
            self.ever_toppled = np.zeros(n, dtype=bool)
            self.ever_fell = np.zeros(n, dtype=bool)
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        self.steps += 1
        if self.available:
            self._measure()
        return result

    def _measure(self):
        poses = self._poses()
        if poses is None or poses.shape != self.initial_pose.shape:
            raise RuntimeError("Distractor number or shape changed mid-episode")
        self.pose_trace.append(poses.copy())
        self.latest_pose = poses
        displacement = np.linalg.norm(poses[:, :3] - self.initial_pose[:, :3], axis=-1)
        rotation = self._rotation_angle(self.initial_pose[:, 3:], poses[:, 3:])
        self.peak_translation = np.maximum(self.peak_translation, displacement)
        self.peak_rotation = np.maximum(self.peak_rotation, rotation)

        quat = poses[:, 3:]
        quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-9)
        upright = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        self.ever_toppled |= upright < math.cos(math.radians(self.topple_angle_deg))
        self.ever_fell |= poses[:, 2] < self.table_height_m - self.fall_below_table_m

    def save_pose_trace(self, path: Path) -> bool:
        """Save [T+1,N,7] trace for future chunk-level label generation."""
        if not self.available or not self.pose_trace:
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            distractor_poses=np.asarray(self.pose_trace, dtype=np.float32),
            object_sizes_m=(self.object_sizes_m.astype(np.float32)
                            if self.object_sizes_m is not None else np.array([], dtype=np.float32)),
            dt_steps=np.int32(self.steps),
        )
        return True

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable disturbance stats and score parameters."""
        if not self.available:
            return {"steps": self.steps, "has_data": False, "continuous_dr": None, "d_max": None}
        residual_translation = np.linalg.norm(
            self.latest_pose[:, :3] - self.initial_pose[:, :3], axis=-1
        )
        residual_rotation = self._rotation_angle(
            self.initial_pose[:, 3:], self.latest_pose[:, 3:]
        )
        n = len(self.initial_pose)
        result = {
            "steps": self.steps,
            "has_data": True,
            "num_distractors": n,
            "peak_translation_m": self.peak_translation.tolist(),
            "residual_translation_m": residual_translation.tolist(),
            "peak_rotation_rad": self.peak_rotation.tolist(),
            "residual_rotation_rad": residual_rotation.tolist(),
            "ever_toppled": self.ever_toppled.tolist(),
            "ever_fell_below_table": self.ever_fell.tolist(),
            "max_peak_translation_m": float(np.max(self.peak_translation)) if n else 0.0,
            "max_peak_rotation_rad": float(np.max(self.peak_rotation)) if n else 0.0,
            "characteristic_sizes_m": None if self.object_sizes_m is None else self.object_sizes_m.tolist(),
            "translation_weight": self.translation_weight,
            "rotation_weight": self.rotation_weight,
            "topple_angle_deg": self.topple_angle_deg,
            "fall_below_table_m": self.fall_below_table_m,
        }
        if self.object_sizes_m is not None:
            result.update(continuous_disturbance_score(
                self.peak_translation, self.peak_rotation, self.object_sizes_m,
                translation_weight=self.translation_weight,
                rotation_weight=self.rotation_weight,
            ))
        else:
            result.update(continuous_dr=None, d_max=None, per_object_disturbance=None)
        return result
