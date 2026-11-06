"""Dense upright scene with wider spawn regions for low objects.

The six distractors retain the sizes and shapes of tight_upright_clutter.
Low boxes can exchange positions within each row. Tall objects stay in safer
zones away from the robot's initial arm pose; the shared sampler rejects
initial overlaps. This ID keeps old tight-upright snapshots replayable.
"""

from dataclasses import dataclass, fields, replace

from mani_skill.utils.registration import register_env

from candid.sim.environment import CANDIDTabletopEnv
from candid.sim.spec import SceneSpec, XYRegion
from envs.candid_tight_upright_env import SCENE as BASE_SCENE


TARGET_ZONE = XYRegion((-0.145, -0.095), (-0.025, 0.025))
GOAL_ZONE = XYRegion((0.08, 0.12), (-0.03, 0.03))
UPPER_BOX_BAND = XYRegion((-0.23, 0.02), (0.085, 0.155))
LOWER_BOX_BAND = XYRegion((-0.23, 0.02), (-0.155, -0.085))
UPPER_CYLINDER_ZONE = XYRegion((-0.16, -0.06), (0.13, 0.18))
UPPER_TALL_BOX_ZONE = XYRegion((-0.04, 0.045), (0.075, 0.15))
LOWER_CYLINDER_ZONE = XYRegion((-0.02, 0.05), (-0.13, -0.075))

NAMES = (
    "upper_blue_box",
    "upper_yellow_cylinder",
    "upper_purple_box",
    "lower_green_box",
    "lower_white_box",
    "lower_orange_cylinder",
)
REGIONS = (
    UPPER_BOX_BAND,
    UPPER_CYLINDER_ZONE,
    UPPER_TALL_BOX_ZONE,
    LOWER_BOX_BAND,
    LOWER_BOX_BAND,
    LOWER_CYLINDER_ZONE,
)
FIXED_XY = (
    (-0.21, 0.12),
    (-0.12, 0.147),
    (0.012, 0.11),
    (-0.21, -0.12),
    (-0.105, -0.12),
    (0.012, -0.10),
)


@dataclass(frozen=True)
class DenseSceneSpec(SceneSpec):
    max_nearest_surface_gap_m: float = 0.08


_base_scene = replace(
    BASE_SCENE,
    name="tight_upright_diverse",
    target=replace(BASE_SCENE.target, region=TARGET_ZONE),
    distractors=tuple(
        replace(obj, name=NAMES[index], region=REGIONS[index],
                fixed_xy=FIXED_XY[index])
        for index, obj in enumerate(BASE_SCENE.distractors)
    ),
    goal_region=GOAL_ZONE,
    max_layout_attempts=300,
)
SCENE = DenseSceneSpec(**{
    field.name: getattr(_base_scene, field.name) for field in fields(SceneSpec)
})


@register_env("CANDIDTightUprightDiverse-v1", max_episode_steps=500)
class CANDIDTightUprightDiverseEnv(CANDIDTabletopEnv):
    SCENE = SCENE
