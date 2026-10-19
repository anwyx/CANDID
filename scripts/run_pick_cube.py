import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import envs.pick_cube_clutter
from planners.pick_cube_lift_place import solve as run_planner
from mani_skill.utils.wrappers.record import RecordEpisode


def main():
    output_dir = Path("outputs/pick_cube_clutter")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "PickCubeClutter-v1",
        robot_uids="xarm6_robotiq",
        obs_mode="none",
        control_mode="pd_joint_pos",
        sim_backend="cpu",
        render_backend="cpu",
        render_mode="rgb_array",
    )

    env = RecordEpisode(
        env,
        output_dir=str(output_dir),
        trajectory_name="pick_cube_clutter",
        save_video=True,
        save_trajectory=True,
        save_on_reset=False,
    )

    try:
        result = run_planner(env, seed=0)

        base = env.unwrapped

        if result == -1:
            print("Motion planning failed")
        else:
            info = result[-1]

            print("Success:", info["success"].item())
            print("Is grasped:", info["is_grasped"].item())
            print("Is object placed:", info["is_obj_placed"].item())
            print("Is robot static:", info["is_robot_static"].item())

            cube_pos = base.cube.pose.p[0]
            goal_pos = base.goal_site.pose.p[0]

            print("Cube position:", cube_pos.tolist())
            print("Goal position:", goal_pos.tolist())
            print("Distance:", torch.linalg.norm(cube_pos - goal_pos).item())

        env.flush_trajectory()
        env.flush_video()
    finally:
        env.close()


if __name__ == "__main__":
    main()
