"""Dense xArm6 tabletop scene with six movable primitive distractors.

The target remains a 4 cm cube so the shared top-down pick-and-place planner
can collect demonstrations. Each object has its own small spawn region: this
keeps the scene crowded while the shared layout sampler prevents initial
overlaps. Object count and dimensions are fixed within this SceneSpec.
"""

from mani_skill.utils.registration import register_env

from candid.sim.environment import CANDIDTabletopEnv
from candid.sim.spec import ObjectSpec, SceneSpec, XYRegion


TARGET_ZONE = XYRegion((-0.14, -0.10), (-0.02, 0.02))
GOAL_ZONE = XYRegion((0.08, 0.12), (-0.03, 0.03))

SCENE = SceneSpec(
    name="dense_primitive_clutter",
    target=ObjectSpec(
        "target_cube", "box", TARGET_ZONE, (-0.12, 0.0),
        size=(0.04, 0.04, 0.04), color=(1.0, 0.1, 0.1, 1.0),
    ),
    distractors=(
        ObjectSpec(
            "upper_left_box", "box",
            XYRegion((-0.20, -0.17), (0.075, 0.105)), (-0.185, 0.09),
            size=(0.035, 0.035, 0.09), color=(0.2, 0.5, 1.0, 1.0),
        ),
        ObjectSpec(
            "upper_middle_cylinder", "cylinder",
            XYRegion((-0.12, -0.09), (0.09, 0.12)), (-0.105, 0.105),
            size=(0.04, 0.04, 0.075), color=(1.0, 0.8, 0.15, 1.0),
        ),
        ObjectSpec(
            "upper_right_flat_box", "box",
            XYRegion((-0.03, 0.0), (0.075, 0.105)), (-0.015, 0.09),
            size=(0.06, 0.04, 0.025), color=(0.7, 0.3, 0.8, 1.0),
        ),
        ObjectSpec(
            "lower_left_cube", "box",
            XYRegion((-0.20, -0.17), (-0.105, -0.075)), (-0.185, -0.09),
            size=(0.035, 0.035, 0.04), color=(0.1, 0.7, 0.3, 1.0),
        ),
        ObjectSpec(
            "lower_middle_tall_cylinder", "cylinder",
            XYRegion((-0.12, -0.09), (-0.12, -0.09)), (-0.105, -0.105),
            size=(0.035, 0.035, 0.11), color=(0.9, 0.9, 0.9, 1.0),
        ),
        ObjectSpec(
            "lower_right_sphere", "sphere",
            XYRegion((-0.03, 0.0), (-0.105, -0.075)), (-0.015, -0.09),
            size=(0.04, 0.04, 0.04), color=(1.0, 0.45, 0.15, 1.0),
        ),
    ),
    goal_region=GOAL_ZONE,
    goal_fixed_xy=(0.10, 0.0),
)


@register_env("CANDIDDenseClutter-v1", max_episode_steps=500)
class CANDIDDenseClutterEnv(CANDIDTabletopEnv):
    SCENE = SCENE
