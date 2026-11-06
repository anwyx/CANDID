"""YCB SceneSpec. No custom _load_scene, logging, or data-writing methods.

YCB assets required: python -m mani_skill.utils.download_asset ycb -y
The generic top-down cube planner is NOT validated for strawberry grasps.
"""
from mani_skill.utils.registration import register_env
from candid.sim.environment import CANDIDTabletopEnv
from candid.sim.spec import SceneSpec, ObjectSpec, XYRegion

FRUIT_TARGET=XYRegion((-.145,-.095),(-.035,.035))
FRUIT_CLUTTER=XYRegion((-.235,.125),(-.195,.195))
GOAL=XYRegion((.080,.120),(-.04,.04))

SCENE=SceneSpec(
    name="ycb_fruit",
    target=ObjectSpec("target_strawberry","ycb",FRUIT_TARGET,(-.12,0),asset_id="012_strawberry"),
    distractors=(
        ObjectSpec("distractor_cleanser","ycb",FRUIT_CLUTTER,(-.155,.14),asset_id="021_bleach_cleanser"),
        ObjectSpec("distractor_apple","ycb",FRUIT_CLUTTER,(-.035,.14),asset_id="013_apple"),
        ObjectSpec("distractor_lemon","ycb",FRUIT_CLUTTER,(-.035,-.14),asset_id="014_lemon"),
        ObjectSpec("distractor_peach","ycb",FRUIT_CLUTTER,(-.155,-.14),asset_id="015_peach"),
    ),
    goal_region=GOAL,goal_fixed_xy=(.10,0.0),
    max_layout_attempts=120,
)

@register_env("CANDIDYCBFruit-v1",max_episode_steps=500)
class CANDIDYCBFruitEnv(CANDIDTabletopEnv):
    SCENE=SCENE
