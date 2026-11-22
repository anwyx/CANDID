"""Six oversized movable distractors close to a small cube pick-and-place task.

The two cylinders have broad bases so they can settle upright on the table.
Each object's XY region is sampled independently; dimensions and object count
are fixed for this scene.
"""

from mani_skill.utils.registration import register_env

from candid.sim.environment import CANDIDTabletopEnv
from candid.sim.spec import ObjectSpec, SceneSpec, XYRegion


TARGET_ZONE = XYRegion((-0.135, -0.105), (-0.015, 0.015))
GOAL_ZONE = XYRegion((0.085, 0.115), (-0.02, 0.02))

SCENE = SceneSpec(
    name="large_primitive_clutter",
    target=ObjectSpec(
        "target_cube", "box", TARGET_ZONE, (-0.12, 0.0),
        size=(0.035, 0.035, 0.035), color=(1.0, 0.1, 0.1, 1.0),
    ),
    distractors=(
        ObjectSpec(
            "upper_left_box", "box",
            XYRegion((-0.235, -0.205), (0.105, 0.13)), (-0.22, 0.117),
            size=(0.075, 0.07, 0.075), color=(0.2, 0.5, 1.0, 1.0),
        ),
        ObjectSpec(
            "upper_middle_cylinder", "cylinder",
            XYRegion((-0.115, -0.085), (0.095, 0.12)), (-0.10, 0.107),
            size=(0.08, 0.08, 0.09), color=(1.0, 0.8, 0.15, 1.0),
        ),
        ObjectSpec(
            "upper_right_box", "box",
            XYRegion((0.025, 0.055), (0.095, 0.12)), (0.04, 0.107),
            size=(0.10, 0.075, 0.05), color=(0.7, 0.3, 0.8, 1.0),
        ),
        ObjectSpec(
            "lower_left_box", "box",
            XYRegion((-0.235, -0.205), (-0.13, -0.105)), (-0.22, -0.117),
            size=(0.075, 0.07, 0.08), color=(0.1, 0.7, 0.3, 1.0),
        ),
        ObjectSpec(
            "lower_middle_cylinder", "cylinder",
            XYRegion((-0.115, -0.085), (-0.12, -0.095)), (-0.10, -0.107),
            size=(0.08, 0.08, 0.10), color=(0.9, 0.9, 0.9, 1.0),
        ),
        ObjectSpec(
            "lower_right_sphere", "sphere",
            XYRegion((0.025, 0.055), (-0.12, -0.095)), (0.04, -0.107),
            size=(0.075, 0.075, 0.075), color=(1.0, 0.45, 0.15, 1.0),
        ),
    ),
    goal_region=GOAL_ZONE,
    goal_fixed_xy=(0.10, 0.0),
)


@register_env("CANDIDLargeClutter-v1", max_episode_steps=500)
class CANDIDLargeClutterEnv(CANDIDTabletopEnv):
    SCENE = SCENE
