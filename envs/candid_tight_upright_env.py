"""Tight clutter: six movable neighbors, including three upright slim objects.

This is a hard scene for collecting successful and disturbed pick-and-place
rollouts. The target stays a 4 cm cube so the shared xArm6 planner can grasp it.
Object sizes and count are fixed; the target, goal, XY positions, and yaw are
sampled from the regions below on each reset.
"""

from mani_skill.utils.registration import register_env

from candid.sim.environment import CANDIDTabletopEnv
from candid.sim.spec import ObjectSpec, SceneSpec, XYRegion


TARGET_ZONE = XYRegion((-0.135, -0.105), (-0.015, 0.015))
GOAL_ZONE = XYRegion((0.085, 0.115), (-0.025, 0.025))

SCENE = SceneSpec(
    name="tight_upright_clutter",
    target=ObjectSpec(
        "target_cube", "box", TARGET_ZONE, (-0.12, 0.0),
        size=(0.04, 0.04, 0.04), color=(1.0, 0.1, 0.1, 1.0),
    ),
    distractors=(
        ObjectSpec(
            "upper_left_box", "box",
            XYRegion((-0.25, -0.22), (0.13, 0.155)), (-0.235, 0.142),
            size=(0.065, 0.06, 0.055), color=(0.2, 0.5, 1.0, 1.0),
        ),
        ObjectSpec(
            "upper_middle_slim_cylinder", "cylinder",
            XYRegion((-0.135, -0.105), (0.135, 0.16)), (-0.12, 0.147),
            size=(0.05, 0.05, 0.12), color=(1.0, 0.8, 0.15, 1.0),
        ),
        ObjectSpec(
            "upper_right_slim_box", "box",
            XYRegion((0.0, 0.025), (0.07, 0.095)), (0.012, 0.082),
            size=(0.05, 0.055, 0.115), color=(0.7, 0.3, 0.8, 1.0),
        ),
        ObjectSpec(
            "lower_left_box", "box",
            XYRegion((-0.24, -0.21), (-0.16, -0.135)), (-0.225, -0.147),
            size=(0.065, 0.06, 0.065), color=(0.1, 0.7, 0.3, 1.0),
        ),
        ObjectSpec(
            "lower_middle_box", "box",
            XYRegion((-0.135, -0.105), (-0.16, -0.135)), (-0.12, -0.147),
            size=(0.065, 0.06, 0.05), color=(0.9, 0.9, 0.9, 1.0),
        ),
        ObjectSpec(
            "lower_right_slim_cylinder", "cylinder",
            XYRegion((0.0, 0.025), (-0.10, -0.075)), (0.012, -0.087),
            size=(0.055, 0.055, 0.115), color=(1.0, 0.45, 0.15, 1.0),
        ),
    ),
    goal_region=GOAL_ZONE,
    goal_fixed_xy=(0.10, 0.0),
    max_layout_attempts=120,
)


@register_env("CANDIDTightUpright-v1", max_episode_steps=500)
class CANDIDTightUprightEnv(CANDIDTabletopEnv):
    SCENE = SCENE
