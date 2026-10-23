"""EXAMPLE: teammate scene = objects + their spawn regions, nothing more.

The shared base handles actor creation, seeded sampling, settling, observation,
reward, and success evaluation. Recording is entirely outside this file.
"""
from mani_skill.utils.registration import register_env
from candid.sim.environment import CANDIDTabletopEnv
from candid.sim.spec import SceneSpec, ObjectSpec, XYRegion

TARGET_ZONE=XYRegion((-0.145,-0.095),(-0.035,0.035))
CLUTTER_ZONE=XYRegion((-0.180,-0.040),(-0.095,0.095))
GOAL_ZONE=XYRegion((0.080,0.120),(-0.040,0.040))

SCENE=SceneSpec(
    name="baseline_cubes",
    target=ObjectSpec("target_cube","box",TARGET_ZONE,(-0.12,0.0),
                      size=(0.04,0.04,0.04),color=(1.0,0.1,0.1,1.0)),
    distractors=(
        ObjectSpec("distractor_white","box",CLUTTER_ZONE,(-.155,.078),size=(.04,.04,.04),color=(.95,.95,.95,1.0)),
        ObjectSpec("distractor_green","box",CLUTTER_ZONE,(-.035,.071),size=(.04,.04,.04),color=(.1,.75,.25,1.0)),
        ObjectSpec("distractor_yellow","box",CLUTTER_ZONE,(-.035,-.077),size=(.04,.04,.04),color=(1.0,.85,.15,1.0)),
        ObjectSpec("distractor_blue","box",CLUTTER_ZONE,(-.155,-.090),size=(.04,.04,.04),color=(.2,.5,1.0,1.0)),
    ),
    goal_region=GOAL_ZONE,goal_fixed_xy=(.1,0.0),
)

@register_env("CANDIDBaseline-v1",max_episode_steps=500)
class CANDIDBaselineEnv(CANDIDTabletopEnv):
    SCENE=SCENE
