from mani_skill.utils.registration import register_env
from candid.sim.environment import CANDIDTabletopEnv
from candid.sim.spec import SceneSpec, ObjectSpec, XYRegion


CLUTTER_ZONE = XYRegion(
    (-0.240, -0.030),
    (-0.140, 0.140)
)


TARGET_ZONE = XYRegion(
    (-0.124, -0.116),
    (-0.004, 0.004)
)

GOAL_ZONE = XYRegion(
    (0.090, 0.110),
    (-0.020, 0.020)
)


SCENE = SceneSpec(
    name="dense_six_boxes",

    target=ObjectSpec(
        "target_cube",
        "box",
        TARGET_ZONE,
        (-0.120, 0.0),
        size=(0.040, 0.040, 0.040),
        color=(1.0, 0.1, 0.1, 1.0)
    ),

    distractors=(
        ObjectSpec(
            "tall_upper_left",
            "box",
            CLUTTER_ZONE,
            (-0.190, 0.070),
            size=(0.035, 0.035, 0.155),
            color=(0.65, 0.40, 0.90, 1.0)
        ),

        ObjectSpec(
            "cube_upper_middle",
            "box",
            CLUTTER_ZONE,
            (-0.125, 0.068),
            size=(0.040, 0.040, 0.040),
            color=(0.20, 0.75, 0.35, 1.0)
        ),

        ObjectSpec(
            "tall_upper_right",
            "box",
            CLUTTER_ZONE,
            (-0.055, 0.073),
            size=(0.035, 0.032, 0.135),
            color=(0.95, 0.62, 0.20, 1.0)
        ),

        ObjectSpec(
            "cube_lower_left",
            "box",
            CLUTTER_ZONE,
            (-0.187, -0.070),
            size=(0.040, 0.040, 0.040),
            color=(0.20, 0.60, 1.0, 1.0)
        ),

        ObjectSpec(
            "tall_lower_middle",
            "box",
            CLUTTER_ZONE,
            (-0.118, -0.073),
            size=(0.035, 0.035, 0.145),
            color=(0.95, 0.80, 0.25, 1.0)
        ),

        ObjectSpec(
            "cube_lower_right",
            "box",
            CLUTTER_ZONE,
            (-0.051, -0.070),
            size=(0.040, 0.040, 0.040),
            color=(0.92, 0.43, 0.52, 1.0)
        ),
    ),

    goal_region=GOAL_ZONE,
    goal_fixed_xy=(0.100, 0.0),

    min_surface_gap=0.003,
    max_layout_attempts=120
)


@register_env("CANDIDDenseClutter-v1", max_episode_steps=500)
class CANDIDDenseClutterEnv(CANDIDTabletopEnv):
    SCENE = SCENE