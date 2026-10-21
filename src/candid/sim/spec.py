"""Team-facing scene descriptions. No ManiSkill, gym, or GPU imports.

Members only need to describe WHAT objects exist and WHERE they may spawn.
New physics assets may extend ActorFactory without changing recording code.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass(frozen=True)
class XYRegion:
    x: tuple[float, float]
    y: tuple[float, float]

    def sample(self, rng):
        return (float(rng.uniform(*self.x)), float(rng.uniform(*self.y)))


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    kind: Literal["box", "cylinder", "sphere", "ycb"]
    region: XYRegion
    fixed_xy: tuple[float, float]
    # box: width/length/height; cylinder: diameter/diameter/height;
    # sphere: diameter/diameter/diameter. YCB dimensions inferred from mesh.
    size: tuple[float, float, float] | None = None
    asset_id: str | None = None
    color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    fixed_yaw: float = 0.0

    def __post_init__(self):
        if self.kind == "ycb" and not self.asset_id:
            raise ValueError(f"{self.name}: YCB requires asset_id, e.g. 013_apple")
        if self.kind != "ycb" and (self.size is None or any(s <= 0 for s in self.size)):
            raise ValueError(f"{self.name}: {self.kind} requires positive size in metres")
        if self.kind in ("sphere", "cylinder") and self.size is not None:
            if abs(self.size[0] - self.size[1]) > 1e-8:
                raise ValueError(f"{self.name}: diameter X and Y must match")


@dataclass(frozen=True)
class SceneSpec:
    name: str
    target: ObjectSpec
    distractors: tuple[ObjectSpec, ...]
    goal_region: XYRegion
    goal_fixed_xy: tuple[float, float]
    robot_base: tuple[float, float, float] = (-0.522, 0.0, 0.0)
    goal_marker_radius: float = 0.06
    goal_xy_tolerance: float = 0.045
    goal_height_tolerance: float = 0.025
    min_surface_gap: float = 0.003
    goal_landing_gap: float = 0.003
    max_layout_attempts: int = 100
    max_position_samples: int = 1000
    camera_eye: tuple[float, float, float] = (0.23, 0.35, 0.65)
    camera_look_at: tuple[float, float, float] = (-0.06, 0.0, 0.04)
    human_camera_eye: tuple[float, float, float] = (0.52, 0.65, 0.58)
    human_camera_look_at: tuple[float, float, float] = (-0.06, 0.0, 0.12)

    def __post_init__(self):
        if not 1 <= len(self.distractors) <= 6:
            raise ValueError("CANDID v1 supports 1..6 distractors")
        names = [self.target.name] + [o.name for o in self.distractors]
        if len(set(names)) != len(names):
            raise ValueError("Object names must be unique within a scene")

    def to_dict(self):
        return asdict(self)
