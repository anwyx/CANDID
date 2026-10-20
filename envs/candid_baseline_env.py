"""CANDID baseline tabletop environment (ManiSkill 3).

Task
----
The xArm6 with a Robotiq gripper picks up ONE red cube and places it in
ONE fixed goal region. Other tabletop objects are movable distractors.
Accidental contacts with distractors are allowed; moving them is NOT an
independent subtask and their displacement is NOT part of the task reward.

Why this file exists
--------------------
All CANDID contributors should use this file as the shared geometric and
simulation convention before adding object assets or domain randomization.
The default reset is deliberately deterministic: no random object poses,
no random robot joint positions, and no random camera poses.

Coordinate convention (metres, ManiSkill world frame)
-----------------------------------------------------
* The tabletop surface is z = 0 (TableSceneBuilder convention).
* The robot is on the -X side; +X points toward the placement goal.
* +/-Y are the two lateral directions; +Z points upward.
* Object pose positions are object CENTRES, not bottom-surface positions.

Sim-to-real observation rule
----------------------------
The agent-visible extra observations below use only the robot's own forward
kinematics and the known task goal. RGB-D sensor data comes from the simulated
camera and can later be replaced with a calibrated physical RGB-D camera.
DO NOT add simulator ground-truth target or distractor poses to _get_obs_extra.
Such poses can be used in an external logger for training labels/evaluation.

Notes
-----
This is a baseline ENVIRONMENT, not a motion planner or a data collector.
It has not been tested end-to-end in a local ManiSkill installation.
"""

from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("CANDIDBaseline-v0", max_episode_steps=250)
class CANDIDBaselineEnv(BaseEnv):
    """Deterministic reference scene for low-disturbance pick-and-place.

    Subclasses can change object geometry or override _initialize_episode
    to randomize positions. Keep the shared coordinate frame, robot pose,
    action/control convention, and goal definition fixed across datasets.
    """

    SUPPORTED_ROBOTS = ["xarm6_robotiq"]
    agent: XArm6Robotiq

    # ================================================================
    # 0. SHARED BASELINE CONFIGURATION -- agree on these as a team.
    # ================================================================
    # Robot base pose. This equals the xArm6 pose in TableSceneBuilder.
    ROBOT_BASE_POS = (-0.522, 0.0, 0.0)
    ROBOT_INIT_QPOS_NOISE = 0.0

    # Picked target (the RED cube). 0.02 m half-side -> 0.04 m side.
    CUBE_HALF_SIZE = 0.020
    TARGET_CUBE_XY = (-0.12, 0.00)
    TARGET_CUBE_YAW = 0.0

    # Goal is a fixed region ON the tabletop, rather than a floating goal.
    # Goal marker is decorative: it has no collision geometry.
    # Keep the place site in front (+X), but within comfortable reach.
    # This places the goal 22 cm ahead of the pickup cube.
    GOAL_XY = (0.10, 0.00)
    GOAL_MARKER_RADIUS = 0.060
    GOAL_XY_TOLERANCE = 0.045
    GOAL_HEIGHT_TOLERANCE = 0.025
    ROBOT_STATIC_THRESHOLD = 0.2

    # Movable clutter: four small cubes for the reference scene only.
    # In later tasks these can be changed to primitives, custom shapes,
    # or YCB assets, while respecting the same world-coordinate region.
    DISTRACTOR_HALF_SIZE = 0.020
    # Two objects near the pickup side, two beside the transfer corridor.
    # The goal itself is kept clear for the place-and-release motion.
    # These centres are deliberately fixed until domain randomization.
    DISTRACTOR_XY = (
        (-0.155,  0.078),
        (-0.035,  0.071),
        (-0.035, -0.077),
        (-0.155, -0.090),
    )
    DISTRACTOR_COLORS = (
        (0.95, 0.95, 0.95, 1.0),  # white
        (0.10, 0.75, 0.25, 1.0),  # green
        (1.00, 0.85, 0.15, 1.0),  # yellow
        (0.20, 0.50, 1.00, 1.0),  # blue
    )

    # Shared region for subsequent RANDOMIZATION (not used at baseline reset).
    # Interpret bounds as object CENTRE x/y; size-dependent clearances still
    # must be checked before accepting sampled positions.
    DISTRACTOR_X_RANGE = (-0.22, 0.03)
    DISTRACTOR_Y_RANGE = (-0.16, 0.16)
    MIN_SURFACE_GAP = 0.010
    # Keep at least 3 cm of free surface clearance around the drawn goal
    # marker (in addition to the distractor bounding-circle radius).
    # Future randomizers should enforce this clearance for every object.
    GOAL_MARKER_FREE_GAP = 0.030

    # Fixed simulated RGB-D camera. Use approximately the same extrinsic
    # pose/intrinsics on the real robot, or calibrate before deployment.
    # Aim cameras closer to the newly shifted manipulation workspace.
    SENSOR_CAM_EYE = (0.23, 0.35, 0.65)
    SENSOR_CAM_LOOK_AT = (-0.06, 0.00, 0.04)
    SENSOR_IMAGE_WIDTH = 256
    SENSOR_IMAGE_HEIGHT = 256
    SENSOR_FOV = np.pi / 2

    # Video-only camera; NOT part of the policy observations.
    HUMAN_CAM_EYE = (0.52, 0.65, 0.58)
    HUMAN_CAM_LOOK_AT = (-0.06, 0.00, 0.12)

    def __init__(self, *args, robot_uids="xarm6_robotiq", **kwargs):
        """Save shared constants and let ManiSkill construct the simulator.

        We intentionally support only the xArm6 Robotiq here: switching
        robots would change reachability, gripper collision geometry,
        motion semantics, and the meaning of a common training scene.
        """
        if robot_uids != "xarm6_robotiq":
            raise ValueError("CANDIDBaseline-v0 uses only xarm6_robotiq")

        self.robot_init_qpos_noise = self.ROBOT_INIT_QPOS_NOISE
        self.cube_half_size = self.CUBE_HALF_SIZE
        self.goal_thresh = self.GOAL_XY_TOLERANCE
        self.goal_radius = self.GOAL_MARKER_RADIUS
        self._validate_baseline_layout()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _validate_baseline_layout(self):
        """Catch accidental geometry errors before simulation starts.

        For the reference cubes, we conservatively approximate each
        footprint by a circle enclosing its square footprint. Later YCB
        or elongated shapes must use their own footprint/collision checks.
        This tests initial overlaps only, NOT robot grasp feasibility.
        """
        target_xy = np.asarray(self.TARGET_CUBE_XY, dtype=float)
        goal_xy = np.asarray(self.GOAL_XY, dtype=float)
        centers = [np.asarray(p, dtype=float) for p in self.DISTRACTOR_XY]
        r_target = np.sqrt(2) * self.CUBE_HALF_SIZE
        r_other = np.sqrt(2) * self.DISTRACTOR_HALF_SIZE
        gap = self.MIN_SURFACE_GAP

        assert len(centers) == len(self.DISTRACTOR_COLORS)
        for i, point in enumerate(centers):
            x, y = point
            if not (self.DISTRACTOR_X_RANGE[0] <= x <= self.DISTRACTOR_X_RANGE[1]):
                raise ValueError(f"Distractor {i} outside shared X range")
            if not (self.DISTRACTOR_Y_RANGE[0] <= y <= self.DISTRACTOR_Y_RANGE[1]):
                raise ValueError(f"Distractor {i} outside shared Y range")
            if np.linalg.norm(point - target_xy) <= r_other + r_target + gap:
                raise ValueError(f"Distractor {i} overlaps the initial target")
            if (
                np.linalg.norm(point - goal_xy)
                <= r_other + self.GOAL_MARKER_RADIUS + self.GOAL_MARKER_FREE_GAP
            ):
                raise ValueError(f"Distractor {i} violates the goal clear area")
            for j in range(i):
                if np.linalg.norm(point - centers[j]) <= 2 * r_other + gap:
                    raise ValueError(f"Distractors {j} and {i} overlap")

    @property
    def _default_sensor_configs(self):
        """Define the RGB-D camera used by both future DP and predictor.

        ManiSkill determines the returned modalities from obs_mode
        (e.g., 'rgbd' or 'pointcloud'). The fixed camera itself is
        just a plausible sim-to-real reference, not a calibrated rig.
        """
        pose = sapien_utils.look_at(
            eye=self.SENSOR_CAM_EYE,
            target=self.SENSOR_CAM_LOOK_AT,
        )
        return [
            CameraConfig(
                "base_camera",
                pose,
                self.SENSOR_IMAGE_WIDTH,
                self.SENSOR_IMAGE_HEIGHT,
                self.SENSOR_FOV,
                0.01,
                10.0,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        """Set a separate, higher-resolution camera for debugging videos."""
        pose = sapien_utils.look_at(
            eye=self.HUMAN_CAM_EYE,
            target=self.HUMAN_CAM_LOOK_AT,
        )
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 10.0)

    def _load_agent(self, options: dict):
        """Load the robot at the team's fixed base pose."""
        super()._load_agent(options, sapien.Pose(p=self.ROBOT_BASE_POS))

    def _load_scene(self, options: dict):
        """Create the table, one target cube, four movable distractors, and goal.

        Everything other than the non-colliding goal marker is a normal
        physics actor. We do NOT disable collision between the gripper,
        picked target, and clutter -- incidental pushing is part of CANDID.
        """
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # The object the robot MUST pick up and place.
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1.0, 0.1, 0.1, 1.0],
            name="target_cube",
            initial_pose=sapien.Pose(
                p=[*self.TARGET_CUBE_XY, self.cube_half_size]
            ),
        )

        # Movable non-target objects, with stable identity/order.
        self.clutter_cubes = []
        for i, (xy, color) in enumerate(
            zip(self.DISTRACTOR_XY, self.DISTRACTOR_COLORS)
        ):
            obj = actors.build_cube(
                self.scene,
                half_size=self.DISTRACTOR_HALF_SIZE,
                color=list(color),
                name=f"distractor_{i}",
                initial_pose=sapien.Pose(
                    p=[xy[0], xy[1], self.DISTRACTOR_HALF_SIZE]
                ),
            )
            self.clutter_cubes.append(obj)

        # The red/white disk has no collision shape, so it cannot
        # accidentally block the target or disturb surrounding objects.
        self.goal_site = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[*self.GOAL_XY, 1e-3]),
        )
        # We intentionally keep the marker visible to the camera: the
        # same marked goal region can be placed on a real tabletop.

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Deterministically reset every object for reproducible rollouts.

        Do not randomize anything here yet. Future subclasses can sample
        layouts, but each candidate trajectory for a fixed scene_id must
        start from exactly the same initial scene.
        """
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # TableSceneBuilder already chooses the xArm6 'rest' joint
            # configuration and the same base pose; restate it here to
            # make our shared convention explicit.
            self.agent.robot.set_pose(sapien.Pose(p=self.ROBOT_BASE_POS))

            # 1. Place the red target cube at the centre of the work area.
            target_p = torch.zeros((b, 3))
            target_p[:, 0] = self.TARGET_CUBE_XY[0]
            target_p[:, 1] = self.TARGET_CUBE_XY[1]
            target_p[:, 2] = self.cube_half_size
            target_q = self._identity_quaternions(b)
            self.cube.set_pose(Pose.create_from_pq(target_p, target_q))
            self._reset_object_velocity(self.cube, b)

            # 2. Place four distractors around the target-to-goal region.
            for obj, xy in zip(self.clutter_cubes, self.DISTRACTOR_XY):
                p = torch.zeros((b, 3))
                p[:, 0] = xy[0]
                p[:, 1] = xy[1]
                p[:, 2] = self.DISTRACTOR_HALF_SIZE
                obj.set_pose(Pose.create_from_pq(p, self._identity_quaternions(b)))
                self._reset_object_velocity(obj, b)

            # 3. Put the non-colliding target marker at a fixed table XY.
            goal_p = torch.zeros((b, 3))
            goal_p[:, 0] = self.GOAL_XY[0]
            goal_p[:, 1] = self.GOAL_XY[1]
            goal_p[:, 2] = 1e-3
            # Follow the marker orientation convention in the given file.
            goal_q = torch.tensor(euler2quat(0, np.pi / 2, 0), dtype=torch.float32)
            self.goal_site.set_pose(Pose.create_from_pq(goal_p, goal_q))

    @staticmethod
    def _reset_object_velocity(obj, batch_size: int):
        """Clear any residual object motion from the previous episode.

        Resetting position alone is not enough when collecting repeated
        contact-heavy rollouts: old linear/angular velocities must not
        leak into a newly reset scene.
        """
        zeros = torch.zeros((batch_size, 3))
        obj.set_linear_velocity(zeros)
        obj.set_angular_velocity(zeros)

    @staticmethod
    def _identity_quaternions(batch_size: int):
        """Return batched (w, x, y, z) quaternions for level cube poses."""
        q = torch.zeros((batch_size, 4))
        q[:, 0] = 1.0
        return q

    def _get_obs_extra(self, info: dict):
        """Return ONLY extra information available on a real robot.

        tcp_pose: obtainable from robot joint encoders + forward kinematics.
        goal_pos: predefined/visually marked and calibrated in the world frame.

        DO NOT include 'is_grasped', cube.pose, clutter poses, or any
        contact labels here; these are simulator-derived task signals.
        The BaseEnv's regular 'agent' observation still provides robot
        proprioception. For learned policies use obs_mode='rgbd' or
        'pointcloud', not a ground-truth object-state observation mode.
        """
        return dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )

    def get_distractor_poses_for_logging(self):
        """Read SIMULATOR GROUND TRUTH for labels/evaluation ONLY.

        Shape: (num_envs, 4, 7); each row is [x, y, z, qw, qx, qy, qz].
        Never feed this data directly to the DP or predictor. A later
        disturbance_logger.py can compare these values to initial poses.
        """
        return torch.stack(
            [obj.pose.raw_pose for obj in self.clutter_cubes], dim=1
        )

    def evaluate(self):
        """Check the pick-and-place goal, independently of disturbance.

        Success requires: cube XY inside goal tolerance, cube resting
        near tabletop height, gripper released, and robot approximately
        static. Distractor motion is intentionally NOT a failure here.
        """
        cube_p = self.cube.pose.p
        goal_p = self.goal_site.pose.p
        xy_error = torch.linalg.norm(cube_p[:, :2] - goal_p[:, :2], dim=1)
        height_error = torch.abs(cube_p[:, 2] - self.cube_half_size)

        is_obj_placed = (
            (xy_error <= self.goal_thresh)
            & (height_error <= self.GOAL_HEIGHT_TOLERANCE)
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_released = ~is_grasped
        is_robot_static = self.agent.is_static(self.ROBOT_STATIC_THRESHOLD)
        success = is_obj_placed & is_released & is_robot_static

        return dict(
            success=success,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasped=is_grasped,
            is_released=is_released,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        """Standard task-only shaping; intentionally no disturbance cost.

        This can help debug task completion or train a task-only RL
        baseline. Vanilla Diffusion Policy will instead learn from the
        successful demonstration actions collected in this environment.
        """
        tcp_to_cube = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, dim=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_cube)
        reward = reaching_reward + info["is_grasped"].float()

        cube_to_goal_xy = torch.linalg.norm(
            self.cube.pose.p[:, :2] - self.goal_site.pose.p[:, :2], dim=1
        )
        place_reward = 1 - torch.tanh(5 * cube_to_goal_xy)
        reward += place_reward * info["is_grasped"].float()

        # The robot's stillness matters AFTER the cube is placed.
        qvel = self.agent.robot.get_qvel()
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, dim=1))
        reward += static_reward * info["is_obj_placed"].float()
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        """Scale the shaped task reward into roughly [0, 1]."""
        return self.compute_dense_reward(obs, action, info) / 5.0
