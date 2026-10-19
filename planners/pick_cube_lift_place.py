"""xArm6 PickCubeClutter debug planner with lift-transport-lower-release.

Preserves the original reach and grasp stages; avoids the too-low
original TCP goal (z=0.001), which represents a table-level target marker.
"""

import numpy as np
import sapien

from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.examples.motionplanning.xarm6.motionplanner import (
    XArm6RobotiqMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def solve(env: PickCubeEnv, seed=None, debug=False, vis=False):
    print("[DEBUG] Entered CANDID xArm6 pick_cube_debug.solve()", flush=True)
    env.reset(seed=seed)
    base = env.unwrapped
    print(f"[DEBUG] robot_uids={base.robot_uids} seed={seed}", flush=True)
    if base.robot_uids != "xarm6_robotiq":
        raise ValueError(f"This planner requires xarm6_robotiq; got {base.robot_uids}")

    initial_clutter = [
        obj.pose.p[0].detach().cpu().numpy().copy()
        for obj in base.clutter_cubes
    ]

    def xyz(actor):
        return actor.pose.p[0].detach().cpu().numpy().copy()

    def report(stage, motion_result=None):
        info = base.evaluate()
        cube_xyz = xyz(base.cube)
        goal_xyz = xyz(base.goal_site)
        tcp_xyz = base.agent.tcp_pose.p[0].detach().cpu().numpy().copy()
        delta = [
            float(np.linalg.norm(xyz(obj) - p0))
            for obj, p0 in zip(base.clutter_cubes, initial_clutter)
        ]
        status = "not checked" if motion_result is None else (
            "PLANNING FAILED" if isinstance(motion_result, (int, np.integer))
            and motion_result == -1 else "path executed"
        )
        flags = {
            key: bool(info[key].item())
            for key in ("is_grasped", "is_obj_placed", "is_robot_static", "success")
        }
        print(f"\n[{stage}] motion={status}", flush=True)
        print(f"  cube_xyz={np.round(cube_xyz, 4)}", flush=True)
        print(f"  tcp_xyz ={np.round(tcp_xyz, 4)}", flush=True)
        print(f"  goal_xyz={np.round(goal_xyz, 4)}", flush=True)
        print(f"  cube_goal_distance={np.linalg.norm(cube_xyz - goal_xyz):.4f} m", flush=True)
        print(f"  flags={flags}", flush=True)
        print(f"  clutter_displacements_m={np.round(delta, 4).tolist()}", flush=True)

    report("INITIAL")
    print("[DEBUG] Initializing XArm6RobotiqMotionPlanningSolver...", flush=True)
    planner = XArm6RobotiqMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    try:
        finger_length = 0.025
        obb = get_actor_obb(base.cube)
        approaching = np.array([0, 0, -1])
        target_closing = (
            base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
            .cpu().numpy()
        )
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=finger_length,
        )
        grasp_pose = base.agent.build_grasp_pose(
            approaching, grasp_info["closing"], base.cube.pose.sp.p
        )
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        print(f"\n[POSE] reach_tcp={np.round(reach_pose.p, 4)}", flush=True)
        print(f"[POSE] grasp_tcp={np.round(grasp_pose.p, 4)}", flush=True)

        print("\n[STEP 1] Reach: RRTStar", flush=True)
        res = planner.move_to_pose_with_RRTStar(reach_pose)
        report("AFTER REACH", res)
        if isinstance(res, (int, np.integer)) and res == -1:
            return -1

        print("\n[STEP 2] Grasp approach: screw", flush=True)
        res = planner.move_to_pose_with_screw(grasp_pose)
        report("AFTER GRASP APPROACH", res)
        if isinstance(res, (int, np.integer)) and res == -1:
            return -1

        print("\n[STEP 3] Close gripper", flush=True)
        planner.close_gripper()
        report("AFTER CLOSE GRIPPER")

        # The red/white goal marker lies on the table (z ~= 0.001).
        # Its position is the cube placement target, NOT a TCP target.
        tcp_now = base.agent.tcp_pose.p[0].detach().cpu().numpy().copy()
        cube_now = xyz(base.cube)
        tcp_relative_to_cube = tcp_now - cube_now

        goal_cube = xyz(base.goal_site)
        goal_cube[2] = float(base.cube_half_size)
        place_tcp = goal_cube + tcp_relative_to_cube

        lift_tcp = tcp_now.copy()
        lift_tcp[2] += 0.12
        transport_tcp = place_tcp.copy()
        transport_tcp[2] = lift_tcp[2]

        print(f"\n[POSE] tcp_relative_to_cube={np.round(tcp_relative_to_cube, 4)}", flush=True)
        print(f"[POSE] lift_tcp     ={np.round(lift_tcp, 4)}", flush=True)
        print(f"[POSE] transport_tcp={np.round(transport_tcp, 4)}", flush=True)
        print(f"[POSE] place_tcp    ={np.round(place_tcp, 4)}", flush=True)

        print("\n[STEP 4A] Lift cube vertically: screw", flush=True)
        res = planner.move_to_pose_with_screw(sapien.Pose(lift_tcp, grasp_pose.q))
        report("AFTER LIFT", res)
        if isinstance(res, (int, np.integer)) and res == -1:
            return -1
        if not bool(base.evaluate()["is_grasped"].item()):
            print("[FAIL] Cube was lost during lift", flush=True)
            return -1

        print("\n[STEP 4B] Transport above goal: RRTStar", flush=True)
        res = planner.move_to_pose_with_RRTStar(sapien.Pose(transport_tcp, grasp_pose.q))
        report("AFTER TRANSPORT", res)
        if isinstance(res, (int, np.integer)) and res == -1:
            print("[HINT] If this fails, check whether the goal at x=0.25 is reachable.", flush=True)
            return -1
        if not bool(base.evaluate()["is_grasped"].item()):
            print("[FAIL] Cube was lost during transport", flush=True)
            return -1

        print("\n[STEP 4C] Lower cube onto goal: screw", flush=True)
        res = planner.move_to_pose_with_screw(sapien.Pose(place_tcp, grasp_pose.q))
        report("AFTER LOWER", res)
        if isinstance(res, (int, np.integer)) and res == -1:
            return -1

        print("\n[STEP 4D] Release cube", flush=True)
        res = planner.open_gripper(t=15)
        report("AFTER RELEASE", res)
        return res
    finally:
        planner.close()
        print("[DEBUG] Planner closed", flush=True)
