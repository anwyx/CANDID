"""Generic idle-drift sanity check for every team environment, no planner needed."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"src"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import argparse
import importlib
import gymnasium as gym
from candid.data.monitor import PhysicsTrace

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--env-module",default="envs.candid_baseline_env")
    parser.add_argument("--env-id",default="CANDIDBaseline-v1")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--steps",type=int,default=100)
    parser.add_argument("--max-idle-dr",type=float,default=0.01)
    args=parser.parse_args()
    importlib.import_module(args.env_module)
    env=gym.make(args.env_id,obs_mode="state_dict",control_mode="pd_joint_pos",
                 robot_uids="xarm6_robotiq",reward_mode="dense",sim_backend="cpu")
    wrapped=PhysicsTrace(env)
    try:
        wrapped.reset(seed=args.seed)
        for _ in range(args.steps):
            wrapped.step(None)
        metrics=wrapped.summary()
        print("Idle disturbance:",metrics)
        if metrics["dr_mean"]>args.max_idle_dr:
            raise SystemExit("FAILED: scene drifts without intentional robot motion")
        print("PASSED: idle drift below diagnostic limit")
    finally:
        wrapped.close()
