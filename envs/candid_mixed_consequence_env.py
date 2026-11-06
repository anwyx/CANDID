"""Larger mixed-shape clutter with varied, close placements.

The target remains a 4 cm cube for the shared xArm6 planner. Two tall objects,
a rolling sphere, and two boxes form a movable corridor around the target.
Broad spawn bands and a nearest-neighbor gap limit provide layout diversity
without leaving isolated distractors.
"""

from dataclasses import dataclass

from mani_skill.utils.registration import register_env

from candid.sim.environment import CANDIDTabletopEnv
from candid.sim.spec import ObjectSpec, SceneSpec, XYRegion


@dataclass(frozen=True)
class MixedConsequenceSceneSpec(SceneSpec):
    max_nearest_surface_gap_m: float = 0.045


SCENE = MixedConsequenceSceneSpec(
    name="mixed_consequence_large_dense",
    target=ObjectSpec(
        "target_cube", "box",
        XYRegion((-0.145, -0.095), (-0.025, 0.025)),
        (-0.120, 0.0),
        size=(0.040, 0.040, 0.040), color=(1.0, 0.1, 0.1, 1.0),
    ),
    distractors=(
        ObjectSpec(
            "upper_tall_cylinder", "cylinder",
            XYRegion((-0.220, 0.020), (0.125, 0.185)),
            (-0.145, 0.150),
            size=(0.056, 0.056, 0.155), color=(0.95, 0.76, 0.18, 1.0),
        ),
        ObjectSpec(
            "lower_tall_box", "box",
            XYRegion((-0.220, 0.005), (-0.175, -0.080)),
            (-0.120, -0.120),
            size=(0.052, 0.050, 0.150), color=(0.56, 0.34, 0.86, 1.0),
        ),
        ObjectSpec(
            "upper_rolling_ball", "sphere",
            XYRegion((-0.075, 0.055), (0.085, 0.165)),
            (-0.040, 0.095),
            size=(0.060, 0.060, 0.060), color=(0.20, 0.68, 0.95, 1.0),
        ),
        ObjectSpec(
            "lower_flat_box", "box",
            XYRegion((-0.140, 0.060), (-0.155, -0.055)),
            (-0.025, -0.105),
            size=(0.085, 0.065, 0.045), color=(0.20, 0.75, 0.38, 1.0),
        ),
        ObjectSpec(
            "upper_goal_box", "box",
            XYRegion((-0.015, 0.105), (0.085, 0.175)),
            (0.055, 0.125),
            size=(0.070, 0.070, 0.080), color=(0.95, 0.48, 0.28, 1.0),
        ),
    ),
    goal_region=XYRegion((0.080, 0.120), (-0.030, 0.030)),
    goal_fixed_xy=(0.100, 0.0),
    max_layout_attempts=400,
)


@register_env("CANDIDMixedConsequence-v1", max_episode_steps=500)
class CANDIDMixedConsequenceEnv(CANDIDTabletopEnv):
    SCENE = SCENE
