"""Example showing that a new physical primitive task only needs a SceneSpec."""
from mani_skill.utils.registration import register_env
from candid.sim.environment import CANDIDTabletopEnv
from candid.sim.spec import SceneSpec, ObjectSpec, XYRegion

T=XYRegion((-.145,-.095),(-.035,.035))
C=XYRegion((-.235,.145),(-.175,.175))
G=XYRegion((.080,.120),(-.040,.040))
SCENE=SceneSpec(
    name="primitive_mixed_shapes",
    target=ObjectSpec("target_cube","box",T,(-.12,0),size=(.04,.04,.04),color=(1,.1,.1,1)),
    distractors=(
        ObjectSpec("tall_box","box",C,(-.15,.115),size=(.035,.035,.10)),
        ObjectSpec("flat_box","box",C,(-.01,.115),size=(.08,.05,.025)),
        ObjectSpec("cylinder","cylinder",C,(-.01,-.115),size=(.045,.045,.08)),
        ObjectSpec("ball","sphere",C,(-.15,-.115),size=(.045,.045,.045)),
    ),
    goal_region=G,goal_fixed_xy=(.1,0),
)

@register_env("CANDIDPrimitive-v1",max_episode_steps=500)
class CANDIDPrimitiveEnv(CANDIDTabletopEnv):
    SCENE=SCENE
