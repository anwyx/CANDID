"""CANDID baseline tabletop environment with bounded domain randomization.

Task
----
A fixed xArm6 + Robotiq gripper picks a red cube and places it in a marked
forward goal region. Four independently movable cubes provide clutter. The
robot is allowed to contact them, but does not deliberately rearrange them.

What "baseline" means here
--------------------------
The robot, action interface, cameras, object geometry, and XY SAMPLING REGIONS
are fixed team-wide. Individual scene layouts are NOT fixed: at each reset,
the target, the goal, and the four distractors receive fresh positions; the
target and distractors also receive a random in-plane yaw.

* reset(seed=same_number) -> same geometry layout (subject to simulator version).
* reset(seed=different_number) -> potentially different geometry layout.
* randomize_layout=False -> old deterministic baseline, for regression tests.

All regions are in metres in the ManiSkill WORLD FRAME. The table surface is
z=0. The robot is on -X; positive X points towards the goal. Positions are
object CENTRES, not the bottom of a collision shape.

Sim-to-real observation rule
----------------------------
Only robot proprioception, the known goal, and simulated camera data are
returned to future learners. The simulator's exact distractor/target poses
may be read via the logging API for supervised labels and debugging, never
as privileged DP or predictor features.

Note: this first randomized baseline varies POSITION and YAW, not shape,
number of distractors, object mass/friction, or camera calibration. Add those
through separate environment extensions after this baseline is validated.
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
    """Reference task with fixed randomization boundaries and seeded resets.

    Scene layouts vary within team-agreed regions. Keep the coordinate frame,
    robot base, task-success metric, camera and action conventions unchanged.
    """

    SUPPORTED_ROBOTS = ["xarm6_robotiq"]
    agent: XArm6Robotiq

    # ================================================================
    # 0. SHARED BASELINE CONFIGURATION -- agree on these as a team.
    # ================================================================
    # Robot base pose. This equals the xArm6 pose in TableSceneBuilder.
    ROBOT_BASE_POS = (-0.522, 0.0, 0.0)
    ROBOT_INIT_QPOS_NOISE = 0.0

    # ----------------------------------------------------------------
    # 1. TARGET: red cube to pick, geometry fixed, reset pose randomized.
    # ----------------------------------------------------------------
    CUBE_HALF_SIZE = 0.020               # side length = 0.04 m
    TARGET_CUBE_XY = (-0.12, 0.00)      # original fixed-scene centre
    TARGET_CUBE_YAW = 0.0

    # XY CENTRE sampling region. The robot base never moves.
    # These narrow bounds reduce unexpected IK failures in the first version.
    TARGET_X_RANGE = (-0.145, -0.095)
    TARGET_Y_RANGE = (-0.035, 0.035)

    # ----------------------------------------------------------------
    # 2. GOAL: marked region on the tabletop, randomized within a small box.
    # ----------------------------------------------------------------
    GOAL_XY = (0.10, 0.00)              # original fixed-scene centre
    GOAL_X_RANGE = (0.080, 0.120)
    GOAL_Y_RANGE = (-0.040, 0.040)
    GOAL_MARKER_RADIUS = 0.060
    GOAL_XY_TOLERANCE = 0.045
    GOAL_HEIGHT_TOLERANCE = 0.025
    ROBOT_STATIC_THRESHOLD = 0.2

    # ----------------------------------------------------------------
    # 3. DISTRACTORS: 4 cube ACTORS; reset samples centre XY and yaw.
    # ----------------------------------------------------------------
    DISTRACTOR_HALF_SIZE = 0.020

    # The original checked fixed layout remains available through
    # randomize_layout=False, for debugging and backwards compatibility.
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

    # Shared randomization bounds for the CENTRES of all four distractors.
    # The sampler below checks clearance to EVERY other sampled object.
    DISTRACTOR_X_RANGE = (-0.22, 0.03)
    DISTRACTOR_Y_RANGE = (-0.16, 0.16)

    # Conservative footprint check: a square of half-size h is enclosed
    # in a circle of radius sqrt(2)*h, valid for ANY in-plane yaw.
    MIN_SURFACE_GAP = 0.010
    GOAL_MARKER_FREE_GAP = 0.030
    MAX_POSITION_SAMPLES = 2000

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

    def __init__(
        self, *args, robot_uids="xarm6_robotiq", robot_init_qpos_noise=0.0,
        randomize_layout=True, **kwargs
    ):
        """Configure the fixed robot and opt-in deterministic regression mode.

        Parameters
        ----------
        randomize_layout:
            True (default): sample target/goal/distractor poses on EACH reset.
            False: use the exact former fixed baseline scene at every reset.
        robot_init_qpos_noise:
            Defaults to zero. Keep zero across the team so the initial robot
            pose stays fixed; pose randomization is limited to task objects.
        """
        if robot_uids != "xarm6_robotiq":
            raise ValueError("CANDIDBaseline-v0 uses only xarm6_robotiq")
        if float(robot_init_qpos_noise) != 0.0:
            raise ValueError(
                "The CANDID baseline fixes robot joint initialization; "
                "robot_init_qpos_noise must be 0.0"
            )
        self.robot_init_qpos_noise = 0.0
        self.randomize_layout = bool(randomize_layout)
        self.cube_half_size = self.CUBE_HALF_SIZE
        self.goal_thresh = self.GOAL_XY_TOLERANCE
        self.goal_radius = self.GOAL_MARKER_RADIUS
        self._validate_randomization_configuration()
        self._last_layouts = {}  # env_id -> pose metadata for logging/debugging
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _validate_randomization_configuration(self):
        """Validate constant bounds and the original fixed fallback layout.

        Full collision/IK validation still takes place during rollout.
        The sampling routine conservatively prevents STARTING intersections
        between movable objects and leaves the goal marker neighbourhood free.
        """
        if len(self.DISTRACTOR_XY) != len(self.DISTRACTOR_COLORS):
            raise ValueError("Number of fixed distractors/colors must match")
        for name, bounds in [
            ("target X", self.TARGET_X_RANGE),
            ("target Y", self.TARGET_Y_RANGE),
            ("goal X", self.GOAL_X_RANGE),
            ("goal Y", self.GOAL_Y_RANGE),
            ("distractor X", self.DISTRACTOR_X_RANGE),
            ("distractor Y", self.DISTRACTOR_Y_RANGE),
        ]:
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError(f"Invalid sampling interval for {name}: {bounds}")
        if self.MIN_SURFACE_GAP < 0 or self.GOAL_MARKER_FREE_GAP < 0:
            raise ValueError("Clearance margins must not be negative")
        fixed = self._fixed_layout()
        placed = []
        for xy in fixed["distractors_xy"]:
            if not self._clear_distractor(xy, fixed["target_xy"],
                                           fixed["goal_xy"], placed):
                raise ValueError("Fixed baseline distractors overlap")
            placed.append(xy)

    @staticmethod
    def _sample_xy(rng, x_range, y_range):
        """Sample one XY position from the fixed rectangular region."""
        return (float(rng.uniform(*x_range)), float(rng.uniform(*y_range)))

    @staticmethod
    def _footprint_radius(half_size):
        """Bounding-circle radius of a square horizontal footprint."""
        return np.sqrt(2.0) * float(half_size)

    def _clear_distractor(self, xy, target_xy, goal_xy, placed_xy):
        """Reject intersecting objects and clutter intruding into the goal.

        This uses conservative 2D bounding circles; it is safe for rotated
        cube footprints. Its purpose is initial separation, not collision
        avoidance once the robot starts moving.
        """
        x, y = xy
        if not (self.DISTRACTOR_X_RANGE[0] <= x <= self.DISTRACTOR_X_RANGE[1]):
            return False
        if not (self.DISTRACTOR_Y_RANGE[0] <= y <= self.DISTRACTOR_Y_RANGE[1]):
            return False
        r_d = self._footprint_radius(self.DISTRACTOR_HALF_SIZE)
        r_t = self._footprint_radius(self.CUBE_HALF_SIZE)
        if np.linalg.norm(np.asarray(xy) - np.asarray(target_xy)) < (
            r_d + r_t + self.MIN_SURFACE_GAP
        ):
            return False
        if np.linalg.norm(np.asarray(xy) - np.asarray(goal_xy)) < (
            r_d + self.GOAL_MARKER_RADIUS + self.GOAL_MARKER_FREE_GAP
        ):
            return False
        for previous in placed_xy:
            if np.linalg.norm(np.asarray(xy) - np.asarray(previous)) < (
                2.0 * r_d + self.MIN_SURFACE_GAP
            ):
                return False
        return True

    def _fixed_layout(self):
        """Return the already tested baseline poses for a regression run."""
        return dict(
            target_xy=tuple(self.TARGET_CUBE_XY),
            target_yaw=float(self.TARGET_CUBE_YAW),
            goal_xy=tuple(self.GOAL_XY),
            distractors_xy=[tuple(xy) for xy in self.DISTRACTOR_XY],
            distractors_yaw=[0.0] * len(self.DISTRACTOR_XY),
        )

    def _sample_layout(self, rng):
        """Generate ONE reproducible non-overlapping randomized scene.

        Important: use ManiSkill's seeded _batched_episode_rng[env_id], not
        Python's unseeded global random module. `reset(seed=n)` therefore
        regenerates the same pose layout across repeated runs.
        """
        target_xy = self._sample_xy(rng, self.TARGET_X_RANGE, self.TARGET_Y_RANGE)
        goal_xy = self._sample_xy(rng, self.GOAL_X_RANGE, self.GOAL_Y_RANGE)
        target_yaw = float(rng.uniform(-np.pi, np.pi))
        placed_xy = []
        placed_yaw = []
        for object_index in range(len(self.DISTRACTOR_XY)):
            for _ in range(self.MAX_POSITION_SAMPLES):
                candidate = self._sample_xy(
                    rng, self.DISTRACTOR_X_RANGE, self.DISTRACTOR_Y_RANGE
                )
                if self._clear_distractor(candidate, target_xy, goal_xy, placed_xy):
                    placed_xy.append(candidate)
                    placed_yaw.append(float(rng.uniform(-np.pi, np.pi)))
                    break
            else:
                raise RuntimeError(
                    f"Could not position distractor {object_index} without "
                    "overlap. Increase workspace bounds or sampling budget."
                )
        return dict(
            target_xy=target_xy,
            target_yaw=target_yaw,
            goal_xy=goal_xy,
            distractors_xy=placed_xy,
            distractors_yaw=placed_yaw,
        )

    @staticmethod
    def _yaw_quaternion(yaws, device):
        """Batched planar orientation in SAPIEN/ManiSkill (w,x,y,z) order."""
        yaws = torch.as_tensor(yaws, dtype=torch.float32, device=device)
        q = torch.zeros((len(yaws), 4), dtype=torch.float32, device=device)
        q[:, 0] = torch.cos(yaws / 2.0)
        q[:, 3] = torch.sin(yaws / 2.0)
        return q

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
        """Build fixed ACTORS once; vary their poses at each episode reset.

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
        """Sample NEW poses on reset, but keep the SAME shared XY regions.

        Geometry is never randomized outside the class-level coordinate
        bounds. The same reset seed produces the same scene; different
        seeds provide new pose layouts. No hidden physical parameters are
        randomized. Each actor's identity stays fixed for consistent logs.
        """
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # The TableSceneBuilder resets the arm's joints to xArm6 'rest'.
            # Keep the robot BASE at its team-fixed coordinate on every reset.
            self.agent.robot.set_pose(sapien.Pose(p=self.ROBOT_BASE_POS))

            # Generate ONE layout for EACH selected simulation environment.
            # For GPU/CPU reproducibility the RNG is indexed per env, as
            # recommended in ManiSkill's randomization documentation.
            env_ids = env_idx.detach().cpu().tolist()
            layouts = []
            for scene_idx in env_ids:
                if self.randomize_layout:
                    rng = self._batched_episode_rng[int(scene_idx)]
                    layout = self._sample_layout(rng)
                else:
                    layout = self._fixed_layout()
                self._last_layouts[int(scene_idx)] = layout
                layouts.append(layout)

            # 1. Picked red cube: random XY and in-plane yaw; centre at z=h.
            target_p = torch.tensor(
                [[*layout["target_xy"], self.cube_half_size] for layout in layouts],
                dtype=torch.float32, device=self.device,
            )
            target_q = self._yaw_quaternion(
                [layout["target_yaw"] for layout in layouts], self.device
            )
            self.cube.set_pose(Pose.create_from_pq(target_p, target_q))
            self._reset_object_velocity(self.cube, b)

            # 2. Distractors: new XY/yaw each reset, sampled with clearance.
            for object_i, obj in enumerate(self.clutter_cubes):
                p = torch.tensor(
                    [
                        [*layout["distractors_xy"][object_i], self.DISTRACTOR_HALF_SIZE]
                        for layout in layouts
                    ],
                    dtype=torch.float32, device=self.device,
                )
                q = self._yaw_quaternion(
                    [layout["distractors_yaw"][object_i] for layout in layouts],
                    self.device,
                )
                obj.set_pose(Pose.create_from_pq(p, q))
                self._reset_object_velocity(obj, b)

            # 3. Goal marker: random XY within a fixed forward landing zone.
            goal_p = torch.tensor(
                [[*layout["goal_xy"], 1e-3] for layout in layouts],
                dtype=torch.float32, device=self.device,
            )
            marker_q = torch.tensor(
                euler2quat(0, np.pi / 2, 0),
                dtype=torch.float32, device=self.device,
            ).expand(b, -1)
            self.goal_site.set_pose(Pose.create_from_pq(goal_p, marker_q))

    def get_scene_layout_for_logging(self):
        """Return sampled layout metadata (NOT a policy observation).

        For the current CPU/MPlib runner num_envs=1. Return dict for one
        environment, or a mapping of id -> layout for parallel simulation.
        This is useful to pair an HDF5 episode with the exact sampled scene.
        """
        if len(self._last_layouts) == 1:
            return next(iter(self._last_layouts.values())).copy()
        return {k: v.copy() for k, v in self._last_layouts.items()}

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
