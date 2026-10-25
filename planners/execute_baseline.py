"""Baseline motion-planning execution for CANDIDBaseline-v0.

This file is deliberately similar to ManiSkill's official xArm6 PickCube
motion-planning example.  It performs ONE task only:

    pre-grasp -> grasp -> lift -> transfer -> place -> release -> retreat

A distractor may be bumped while the robot moves or carries the red cube.
The planner does NOT intentionally grasp, relocate, or rearrange distractors.
Distractors are NOT inserted as hard obstacles in MPlib's point-cloud world.

The planner is a DEMONSTRATION GENERATOR, not the learned CANDID policy.
It is allowed to read the simulator's exact cube pose to compute an expert
trajectory.  Do not feed that pose into a future DP/predictor observation.

Required environment control mode: ``pd_joint_pos``.  This is the control
mode expected by ManiSkill's provided XArm6RobotiqMotionPlanningSolver.
Record the actual joint-position + gripper commands; do not later label these
commands as end-effector delta-pose commands without proper conversion.

ManiSkill source examples used as implementation reference:
    mani_skill/examples/motionplanning/xarm6/solutions/pick_cube.py
    mani_skill/examples/motionplanning/xarm6/motionplanner.py
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import sapien
from candid.runner.variants import validate_xyz_offset, waypoint_between

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.xarm6.motionplanner import (
    XArm6RobotiqMotionPlanningSolver,
)


@dataclass
class BaselinePlanConfig:
    """Geometry and execution settings shared by all baseline rollouts.

    Distances are in metres.  Change these only after visually checking the
    nominal scene and checking reachability; do not silently change them in
    different team members' data-collection scripts.
    """

    approach_height: float = 0.065  # Clearance above the target before grasping.
    lift_height: float = 0.120      # Raise the held cube above its grasp pose.
    transfer_clearance: float = 0.080  # Extra clearance above the final place pose.
    retreat_height: float = 0.100  # Raise open gripper after dropping the cube.

    gripper_close_steps: int = 10
    gripper_open_steps: int = 12
    settle_steps: int = 12

    # 0 = direct transfer; +/- values introduce an intermediate lateral
    # waypoint.  This is an intentionally simple, non-random baseline for
    # later waypoint perturbations and demonstration diversity.
    lateral_offset: float = 0.0
    # XYZ offsets in WORLD coordinates. The final grasp and goal stay fixed.
    early_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lift_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    transfer_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    strategy: str = "manual"
    collision_aware: bool = False
    collision_margin_m: float = 0.008

    # Intermediate transfer motion uses RRT*, as in the official examples.
    # Approach/place use the simpler screw motion when possible.
    finger_length: float = 0.025
    min_lifted_cube_height: float = 0.050


@dataclass
class PlanResult:
    """Result returned to the script after a single complete attempt."""

    success: bool
    stage: str
    steps: int
    grasped_after_lift: bool = False
    message: str = ""


def _numpy_position(pose) -> np.ndarray:
    """Convert a ManiSkill batched Pose position to one CPU XYZ vector.

    This planner executes one simulation environment per call (num_envs=1).
    In a GPU-parallel data collector, an individual per-env planner is needed.
    """
    return pose.p[0].detach().cpu().numpy().astype(np.float64)


def _pose(p_xyz, quaternion) -> sapien.Pose:
    """Construct a world-frame TCP pose, keeping gripper orientation."""
    return sapien.Pose(p=np.asarray(p_xyz, dtype=float), q=np.asarray(quaternion, dtype=float))


def _add_distractor_collisions(planner, base_env, margin_m: float):
    """Add only distractors to MPlib as approximate, padded oriented boxes.

    The target cube is deliberately excluded so the robot can grasp it.
    SAPIEN still simulates the real geometries and all contact physics.
    This is a static point-cloud approximation, not a proof of collision freedom.
    """
    if not np.isfinite(margin_m) or not 0.0 <= margin_m <= 0.05:
        raise ValueError("collision_margin_m must be within [0, 0.05]")

    for actor in base_env.distractor_actors:
        mesh = actor.get_first_collision_mesh(to_world_frame=False)
        if mesh is None:
            raise RuntimeError(f"No collision mesh for distractor {actor.name}")
        bounds = np.asarray(mesh.bounding_box.bounds, dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise RuntimeError(f"Invalid collision bounds for {actor.name}")
        centre_local = (bounds[0] + bounds[1]) / 2.0
        extent = bounds[1] - bounds[0] + 2.0 * margin_m
        if np.any(extent <= 0):
            raise RuntimeError(f"Invalid collision extent for {actor.name}")
        world_box_pose = actor.pose.sp * sapien.Pose(p=centre_local)
        planner.add_box_collision(extent, world_box_pose)


def _stage(planner, name: str, tcp_pose: sapien.Pose, method: str, verbose: bool):
    """Plan and EXECUTE a TCP move; stop on an MPlib planning failure.

    Both ManiSkill solver methods return -1 on failure, or the usual
    (obs, reward, terminated, truncated, info) tuple after execution.
    The simulator is responsible for all resulting physical contacts.
    """
    if verbose:
        print(f"  [{name}] planner={method}, target TCP={np.round(tcp_pose.p, 3)}")
    if method == "rrt":
        result = planner.move_to_pose_with_RRTStar(tcp_pose)
    elif method == "screw":
        result = planner.move_to_pose_with_screw(tcp_pose)
    else:
        raise ValueError(f"Unknown motion-planning method: {method}")
    if isinstance(result, (int, np.integer)) and result == -1:
        if verbose:
            print(f"  [{name}] failed to plan")
        return False
    if result is None:
        return False
    return True


def _grasp_pose(base_env, finger_length: float) -> sapien.Pose:
    """Compute the top-down grasp pose using ManiSkill's own XArm6 logic.

    This uses the cube's *simulation GT* only to create teacher actions.
    The final CANDID observation should instead come from RGB-D and robot
    proprioception (already defined in the environment file).
    """
    cube = base_env.cube
    obb = get_actor_obb(cube)
    approaching = np.array([0.0, 0.0, -1.0])
    target_closing = (
        base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
        .detach()
        .cpu()
        .numpy()
    )
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=finger_length,
    )
    # The official XArm6 PickCube solver builds its grasp at the cube centre.
    # This also works with the fixed 4-cm cube in the CANDID reference scene.
    return base_env.agent.build_grasp_pose(
        approaching,
        grasp_info["closing"],
        cube.pose.sp.p,
    )


def _success(base_env) -> bool:
    """Evaluate task completion independently of disturbance."""
    result = base_env.evaluate()["success"]
    return bool(result.detach().cpu().reshape(-1)[0].item())


def solve(
    env,
    seed: Optional[int] = None,
    config: Optional[BaselinePlanConfig] = None,
    vis: bool = False,
    verbose: bool = False,
    reset_env: bool = True,
) -> PlanResult:
    """Reset and execute exactly one CANDID baseline Pick-and-Place attempt.

    Parameters
    ----------
    env:
        ``gym.make('CANDIDBaseline-v0', control_mode='pd_joint_pos', ...)``
        or the same environment wrapped with RecordEpisode / a logger.
    seed:
        Reset seed.  The baseline environment is deterministic, so changing
        this seed alone does NOT create different geometry configurations.
    config:
        Shared baseline motion settings, including optional lateral offset.
    vis:
        Show ManiSkill's interactive viewer.  Turn off on headless servers.
    verbose:
        Print step/stage messages.

    Returns
    -------
    PlanResult:
        Always report the stage reached and the actual environment success.
        An RRT/screw planning failure is not a successful demonstration.
    """
    cfg = config if config is not None else BaselinePlanConfig()
    early_delta = validate_xyz_offset(cfg.early_offset_xyz, "early_offset_xyz")
    lift_delta = validate_xyz_offset(cfg.lift_offset_xyz, "lift_offset_xyz")
    transfer_delta = validate_xyz_offset(cfg.transfer_offset_xyz, "transfer_offset_xyz")

    # Reset BEFORE creating MPlib's planner, so both planners and physics
    # operate from the exact initial state that will be recorded.
    if reset_env:
        env.reset(seed=seed)
    base = env.unwrapped
    if base.num_envs != 1:
        raise ValueError("Baseline planner requires num_envs=1")
    if base.control_mode != "pd_joint_pos":
        raise ValueError("Use control_mode='pd_joint_pos' with the MPlib XArm6 solver")

    stage = "initialization"
    grasped_after_lift = False
    planner = None

    try:
        planner = XArm6RobotiqMotionPlanningSolver(
            env,
            debug=False,
            vis=vis,
            base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=vis,
            print_env_info=False,
        )
        # IMPORTANT: no distractor point cloud is added to MPlib.  The
        # simulation nevertheless keeps distractor collisions ENABLED.
        planner.use_point_cloud = False
        if cfg.collision_aware:
            _add_distractor_collisions(planner, base, cfg.collision_margin_m)

        stage = "open_gripper"
        planner.open_gripper(t=cfg.gripper_open_steps)

        grasp_pose = _grasp_pose(base, cfg.finger_length)
        pregrasp = grasp_pose * sapien.Pose([0, 0, -cfg.approach_height])

        # New: sample a distinct approach corridor BEFORE the common grasp.
        # Apply a limited Z deviation while retaining grasp clearance.
        if np.linalg.norm(early_delta) > 1e-8:
            stage = "early_waypoint"
            early_xyz = np.asarray(pregrasp.p, dtype=float) + early_delta
            # Allow limited negative Z for axis ablation, while keeping
            # the waypoint >=3 cm above the final grasp TCP position.
            early_xyz[2] = max(early_xyz[2], float(grasp_pose.p[2]) + 0.030)
            if not _stage(planner, stage, _pose(early_xyz, pregrasp.q), "rrt", verbose):
                return PlanResult(False, stage, planner.elapsed_steps,
                                  message="early waypoint planning failed")

        stage = "pregrasp"
        if not _stage(planner, stage, pregrasp, "rrt", verbose):
            return PlanResult(False, stage, planner.elapsed_steps, message="pregrasp planning failed")

        stage = "approach"
        if not _stage(planner, stage, grasp_pose, "screw", verbose):
            return PlanResult(False, stage, planner.elapsed_steps, message="grasp approach failed")

        stage = "close_gripper"
        planner.close_gripper(t=cfg.gripper_close_steps)

        stage = "lift"
        lift_pose = grasp_pose * sapien.Pose([0, 0, -cfg.lift_height])
        # New: route the carried object through a varied 3D lift path.
        if np.linalg.norm(lift_delta) > 1e-8:
            stage = "lift_waypoint"
            lift_xyz = waypoint_between(grasp_pose.p, lift_pose.p, lift_delta,
                                        min_z=float(grasp_pose.p[2]) + 0.025)
            if not _stage(planner, stage, _pose(lift_xyz, grasp_pose.q), "rrt", verbose):
                return PlanResult(False, stage, planner.elapsed_steps,
                                  message="lift waypoint planning failed")
            stage = "lift"
        if not _stage(planner, stage, lift_pose, "screw", verbose):
            return PlanResult(False, stage, planner.elapsed_steps, message="lifting failed")

        # A successful pick is required.  If the cube dropped, continuing to
        # the goal with an empty gripper would create a misleading demo.
        grasped_after_lift = bool(base.agent.is_grasping(base.cube)[0].item())
        cube_xyz = _numpy_position(base.cube.pose)
        if not grasped_after_lift or cube_xyz[2] < cfg.min_lifted_cube_height:
            return PlanResult(
                False,
                "check_grasp",
                planner.elapsed_steps,
                grasped_after_lift,
                "target was not successfully lifted",
            )

        # Observe the real displacement between the held cube and TCP.
        # Keeping this offset during transfer makes the final *cube centre*,
        # not the TCP, land on the table-level goal position.
        tcp_xyz = _numpy_position(base.agent.tcp.pose)
        grasp_offset = tcp_xyz - cube_xyz
        goal_xyz = _numpy_position(base.goal_site.pose)
        cube_place_xyz = np.array(
            [goal_xyz[0], goal_xyz[1], base.cube_half_size], dtype=float
        )
        tcp_place_xyz = cube_place_xyz + grasp_offset
        tcp_above_goal_xyz = tcp_place_xyz + [0, 0, cfg.transfer_clearance]
        grasp_quat = grasp_pose.q

        # For later trajectory diversification: setting a nonzero lateral
        # offset creates a single intermediate waypoint, without introducing
        # a new manipulation task or actively relocating a distractor.
        if abs(cfg.lateral_offset) > 1e-6 or np.linalg.norm(transfer_delta) > 1e-8:
            stage = "transfer_waypoint"
            waypoint_xyz = 0.5 * (tcp_xyz + tcp_above_goal_xyz)
            waypoint_xyz[1] += cfg.lateral_offset
            waypoint_xyz[2] = max(tcp_xyz[2], tcp_above_goal_xyz[2])
            if np.linalg.norm(transfer_delta) > 1e-8:
                waypoint_xyz += transfer_delta
                waypoint_xyz[2] = max(waypoint_xyz[2], base.cube_half_size + 0.06)
            if not _stage(planner, stage, _pose(waypoint_xyz, grasp_quat), "rrt", verbose):
                return PlanResult(False, stage, planner.elapsed_steps, grasped_after_lift)

        stage = "transfer"
        if not _stage(planner, stage, _pose(tcp_above_goal_xyz, grasp_quat), "rrt", verbose):
            return PlanResult(False, stage, planner.elapsed_steps, grasped_after_lift)

        stage = "lower"
        if not _stage(planner, stage, _pose(tcp_place_xyz, grasp_quat), "screw", verbose):
            return PlanResult(False, stage, planner.elapsed_steps, grasped_after_lift)

        stage = "release"
        planner.open_gripper(t=cfg.gripper_open_steps)

        stage = "retreat"
        retreat_xyz = tcp_place_xyz + np.array([0.0, 0.0, cfg.retreat_height])
        if not _stage(planner, stage, _pose(retreat_xyz, grasp_quat), "screw", verbose):
            return PlanResult(False, stage, planner.elapsed_steps, grasped_after_lift)

        # Hold the robot in place with the gripper open.  This allows the
        # cube to settle and the 'is_robot_static' success check to pass.
        stage = "settle"
        planner.open_gripper(t=cfg.settle_steps)
        return PlanResult(
            success=_success(base),
            stage="completed",
            steps=planner.elapsed_steps,
            grasped_after_lift=grasped_after_lift,
            message="task-success check is independent of distractor motion",
        )
    finally:
        if planner is not None:
            planner.close()
