"""CANDID YCB-fruit clutter: strawberry pick-and-place among movable objects.

Based on the team's earlier PickFruitClutter-v1 environment. The environment
registers CANDIDYCBFruit-v0 and supports the generic CANDID runner:

    python scripts/run_baseline.py \
      --env-module envs.candid_ycb_fruit_env --env-id CANDIDYCBFruit-v0 \
      --episodes 2 --obs-mode state --output-dir demos/ycb_fruit

Goal: place the YCB strawberry in a non-colliding tabletop goal region while
leaving the bleach cleanser, apple, lemon, and peach undisturbed. Distractors
are dynamic; moving them does NOT, by itself, fail the task.

CANDID integration:
  * Truly continuous XY/yaw randomization using ManiSkill's per-episode RNG.
  * Physics settling before observations and disturbance logging start.
  * Conservative collision-mesh-based initial XY footprint clearances.
  * State observations with robot/target/distractor states for sim-only DP.
  * Stable distractor order, world-pose and size APIs for DisturbanceMonitor.
  * Explicit fixed-scene mode for debugging (--fixed-scene).

The supplied baseline xArm6 planner may ATTEMPT this task through the
'.cube' / '.cube_half_size' compatibility aliases, but has not been adapted
or validated for strawberry grasping. A specialized planner may be necessary.
YCB object assets must be installed in the ManiSkill container.
"""

from __future__ import annotations

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


@register_env("CANDIDYCBFruit-v0", max_episode_steps=500)
class CANDIDYCBFruitEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["xarm6_robotiq"]
    agent: XArm6Robotiq

    ROBOT_BASE_POS = (-0.522, 0.0, 0.0)
    TARGET_MODEL_ID = "012_strawberry"
    DISTRACTOR_MODEL_IDS = (
        "021_bleach_cleanser",
        "013_apple",
        "014_lemon",
        "015_peach",
    )

    TARGET_X_RANGE = (-0.145, -0.095)
    TARGET_Y_RANGE = (-0.035, 0.035)
    GOAL_X_RANGE = (0.080, 0.120)
    GOAL_Y_RANGE = (-0.035, 0.035)

    TARGET_FIXED_XY = (-0.12, 0.00)
    GOAL_FIXED_XY = (0.10, 0.00)
    # Sample continuously across the *entire* clutter workspace (metres).
    # These limits refer to object ORIGINS, not guaranteed mesh extents.
    # The clearance validator below rejects overlapping configurations.
    DISTRACTOR_X_RANGE = (-0.235, 0.065)
    DISTRACTOR_Y_RANGE = (-0.215, 0.215)
    # For --fixed-scene only: deterministic regression layout.
    DISTRACTOR_FIXED_XY = (
        (-0.205, 0.165),
        (0.005, 0.155),
        (0.005, -0.155),
        (-0.205, -0.165),
    )
    INITIAL_FLOOR_CLEARANCE_M = 0.002
    MIN_SURFACE_GAP = 0.005
    GOAL_LANDING_GAP = 0.010
    MAX_LAYOUT_ATTEMPTS = 300
    MAX_POSITION_SAMPLES = 200

    GOAL_MARKER_RADIUS = 0.060  # purely visual, not a solid obstacle
    GOAL_XY_TOLERANCE = 0.045
    GOAL_HEIGHT_TOLERANCE = 0.030
    ROBOT_STATIC_THRESHOLD = 0.2

    SENSOR_CAM_EYE = (0.23, 0.35, 0.65)
    SENSOR_CAM_LOOK_AT = (-0.06, 0.0, 0.05)
    HUMAN_CAM_EYE = (0.52, 0.65, 0.58)
    HUMAN_CAM_LOOK_AT = (-0.06, 0.0, 0.12)

    def __init__(
        self,
        *args,
        robot_uids: str = "xarm6_robotiq",
        robot_init_qpos_noise: float = 0.0,
        randomize_layout: bool = True,
        distractor_x_range: tuple[float, float] | None = None,
        distractor_y_range: tuple[float, float] | None = None,
        settle_steps: int = 40,
        idle_verify_steps: int = 12,
        max_idle_translation_m: float = 0.0015,
        max_idle_rotation_deg: float = 2.0,
        **kwargs,
    ):
        if robot_uids != "xarm6_robotiq":
            raise ValueError("CANDIDYCBFruit-v0 currently supports only xarm6_robotiq")
        if float(robot_init_qpos_noise) != 0.0:
            raise ValueError("Use robot_init_qpos_noise=0 for reproducible CANDID scenes")
        self.robot_init_qpos_noise = 0.0
        self.randomize_layout = bool(randomize_layout)
        self.distractor_x_range = tuple(float(x) for x in (
            distractor_x_range if distractor_x_range is not None else self.DISTRACTOR_X_RANGE
        ))
        self.distractor_y_range = tuple(float(y) for y in (
            distractor_y_range if distractor_y_range is not None else self.DISTRACTOR_Y_RANGE
        ))
        if len(self.distractor_x_range) != 2 or self.distractor_x_range[0] >= self.distractor_x_range[1]:
            raise ValueError("distractor_x_range must be [min_x, max_x]")
        if len(self.distractor_y_range) != 2 or self.distractor_y_range[0] >= self.distractor_y_range[1]:
            raise ValueError("distractor_y_range must be [min_y, max_y]")
        self.settle_steps = int(settle_steps)
        self.idle_verify_steps = int(idle_verify_steps)
        self.max_idle_translation_m = float(max_idle_translation_m)
        self.max_idle_rotation_deg = float(max_idle_rotation_deg)
        if min(self.settle_steps, self.idle_verify_steps) < 0:
            raise ValueError("settle_steps and idle_verify_steps must be nonnegative")
        if self.max_idle_translation_m <= 0 or self.max_idle_rotation_deg <= 0:
            raise ValueError("Idle drift tolerances must be positive")
        self.goal_radius = self.GOAL_MARKER_RADIUS
        self.goal_thresh = self.GOAL_XY_TOLERANCE
        self._last_layouts: dict[int, dict[str, Any]] = {}
        # Filled from actual YCB collision meshes in _after_reconfigure().
        self.target_rest_z: float | None = None
        self.clutter_rest_z: list[float] = []
        self.target_footprint_radius: float | None = None
        self.clutter_footprint_radii: list[float] = []
        self.target_dimensions: list[float] = []
        self.clutter_dimensions: list[list[float]] = []
        self.clutter_size_scales: list[float] = []
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        camera_pose = sapien_utils.look_at(
            eye=self.SENSOR_CAM_EYE, target=self.SENSOR_CAM_LOOK_AT,
        )
        return [CameraConfig("base_camera", camera_pose, 256, 256, np.pi / 2, 0.01, 10.0)]

    @property
    def _default_human_render_camera_configs(self):
        camera_pose = sapien_utils.look_at(
            eye=self.HUMAN_CAM_EYE, target=self.HUMAN_CAM_LOOK_AT,
        )
        return CameraConfig("render_camera", camera_pose, 512, 512, 1.0, 0.01, 10.0)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=self.ROBOT_BASE_POS))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        target_builder = actors.get_actor_builder(
            self.scene, id=f"ycb:{self.TARGET_MODEL_ID}",
        )
        target_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
        self.target_object = target_builder.build(name="target_strawberry")
        self.cube = self.target_object  # Compatibility with default xArm6 baseline planner

        self.clutter_objects = []
        for model_id in self.DISTRACTOR_MODEL_IDS:
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
            builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
            self.clutter_objects.append(builder.build(name=f"distractor_{model_id}"))
        # Compatibility for third-party logging tools expecting the old field.
        self.clutter_cubes = self.clutter_objects

        self.goal_site = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[*self.GOAL_FIXED_XY, 1e-3]),
        )

    @staticmethod
    def _collision_geometry(actor, actor_name: str):
        """Get upright resting height, footprint radius, and [x,y,z] size.

        The footprint radius encloses ALL corners of the local collision-mesh
        XY bounding box relative to the actor origin, so it remains conservative
        for arbitrary yaw rotations even if the origin is off-centre.
        """
        mesh = actor.get_first_collision_mesh(to_world_frame=False)
        if mesh is None:
            raise RuntimeError(f"YCB actor {actor_name} has no collision mesh")
        bounds = np.asarray(mesh.bounding_box.bounds, dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            raise RuntimeError(f"Unexpected collision mesh bounds for {actor_name}: {bounds}")
        xy_corners = np.array([
            [x, y]
            for x in (bounds[0, 0], bounds[1, 0])
            for y in (bounds[0, 1], bounds[1, 1])
        ])
        radius = float(np.linalg.norm(xy_corners, axis=1).max())
        dimensions = (bounds[1] - bounds[0]).tolist()
        rest_z = float(-bounds[0, 2])
        if radius <= 0 or max(dimensions) <= 0:
            raise RuntimeError(f"Invalid YCB geometry for {actor_name}: {dimensions}")
        return rest_z, radius, dimensions

    def _after_reconfigure(self, options: dict):
        (
            self.target_rest_z,
            self.target_footprint_radius,
            self.target_dimensions,
        ) = self._collision_geometry(self.target_object, self.TARGET_MODEL_ID)

        self.clutter_rest_z = []
        self.clutter_footprint_radii = []
        self.clutter_dimensions = []
        self.clutter_size_scales = []
        for model_id, obj in zip(self.DISTRACTOR_MODEL_IDS, self.clutter_objects):
            z, radius, dimensions = self._collision_geometry(obj, model_id)
            self.clutter_rest_z.append(z)
            self.clutter_footprint_radii.append(radius)
            self.clutter_dimensions.append(dimensions)
            self.clutter_size_scales.append(float(max(dimensions)))

        # Original baseline planner computes place-Z using cube_half_size.
        # Here the YCB strawberry's correct resting centre height is used.
        self.cube_half_size = self.target_rest_z

    @staticmethod
    def _sample_xy(rng, x_limits, y_limits):
        return (
            float(rng.uniform(*x_limits)),
            float(rng.uniform(*y_limits)),
        )

    @staticmethod
    def _far_enough(a, b, minimum_distance):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) >= minimum_distance

    def _candidate_is_valid(self, xy, radius, target_xy, goal_xy, occupied):
        if not self._far_enough(
            xy, target_xy,
            radius + self.target_footprint_radius + self.MIN_SURFACE_GAP,
        ):
            return False
        # The goal disk is visual only: reserve physical room for placing the
        # STRAWBERRY, not an arbitrary forbidden zone around the disk radius.
        if not self._far_enough(
            xy, goal_xy,
            radius + self.target_footprint_radius + self.GOAL_LANDING_GAP,
        ):
            return False
        return all(
            self._far_enough(xy, other_xy, radius + other_radius + self.MIN_SURFACE_GAP)
            for other_xy, other_radius in occupied
        )

    def _fixed_layout(self):
        return {
            "target_xy": list(self.TARGET_FIXED_XY),
            "target_yaw": 0.0,
            "goal_xy": list(self.GOAL_FIXED_XY),
            "distractors_xy": [list(xy) for xy in self.DISTRACTOR_FIXED_XY],
            "distractors_yaw": [0.0] * len(self.DISTRACTOR_MODEL_IDS),
        }

    def _sample_layout(self, rng):
        """Randomize all object XY positions uniformly over continuous regions.

        No slots/permutation/jitter.  We use conservative footprint-circle
        rejection sampling to avoid *initial* penetration.  Yaw is sampled
        independently and continuously for each YCB actor.
        """
        for _ in range(self.MAX_LAYOUT_ATTEMPTS):
            target_xy = self._sample_xy(rng, self.TARGET_X_RANGE, self.TARGET_Y_RANGE)
            goal_xy = self._sample_xy(rng, self.GOAL_X_RANGE, self.GOAL_Y_RANGE)
            placed = []
            xy_positions = []
            yaws = []
            # Place larger distractors first for more reliable packing.
            order = sorted(
                range(len(self.DISTRACTOR_MODEL_IDS)),
                key=lambda i: self.clutter_footprint_radii[i], reverse=True,
            )
            for index in order:
                radius = self.clutter_footprint_radii[index]
                for _ in range(self.MAX_POSITION_SAMPLES):
                    xy = self._sample_xy(rng, self.distractor_x_range, self.distractor_y_range)
                    if self._candidate_is_valid(xy, radius, target_xy, goal_xy, placed):
                        placed.append((xy, radius))
                        xy_positions.append((index, list(xy)))
                        yaws.append((index, float(rng.uniform(-np.pi, np.pi))))
                        break
                else:
                    break
            else:
                # Indices, not placement order, define stable model identity.
                xy_positions.sort(key=lambda x: x[0])
                yaws.sort(key=lambda x: x[0])
                return {
                    "target_xy": list(target_xy),
                    "target_yaw": float(rng.uniform(-np.pi, np.pi)),
                    "goal_xy": list(goal_xy),
                    "distractors_xy": [x for _, x in xy_positions],
                    "distractors_yaw": [yaw for _, yaw in yaws],
                }
        raise RuntimeError(
            "Could not sample four non-overlapping YCB objects in the configured workspace. "
            "Increase distractor_x_range / distractor_y_range or decrease object count; "
            "the model collision footprints may be too large."
        )

    @staticmethod
    def _yaw_quaternions(angles, device):
        angles = torch.as_tensor(angles, dtype=torch.float32, device=device)
        return torch.stack(
            [torch.cos(angles / 2), torch.zeros_like(angles),
             torch.zeros_like(angles), torch.sin(angles / 2)],
            dim=-1,
        )

    @staticmethod
    def _reset_actor_velocity(actor, b, device):
        zeros = torch.zeros((b, 3), dtype=torch.float32, device=device)
        actor.set_linear_velocity(zeros)
        actor.set_angular_velocity(zeros)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.agent.robot.set_pose(sapien.Pose(p=self.ROBOT_BASE_POS))

            if self.target_rest_z is None:
                raise RuntimeError("YCB collision geometry not initialized before reset")

            scene_ids = env_idx.detach().cpu().tolist()
            layouts = []
            for scene_id in scene_ids:
                if self.randomize_layout:
                    rng = self._batched_episode_rng[int(scene_id)]
                    layout = self._sample_layout(rng)
                else:
                    layout = self._fixed_layout()
                    occupied = []
                    for index, xy in enumerate(layout["distractors_xy"]):
                        radius = self.clutter_footprint_radii[index]
                        if not self._candidate_is_valid(
                            xy, radius, layout["target_xy"], layout["goal_xy"], occupied,
                        ):
                            raise ValueError(
                                f"Fixed fruit layout has overlapping objects near {self.DISTRACTOR_MODEL_IDS[index]}. "
                                "Use randomized layout or move DISTRACTOR_SLOTS_XY."
                            )
                        occupied.append((xy, radius))
                self._last_layouts[int(scene_id)] = layout
                layouts.append(layout)

            p = torch.tensor(
                [[*layout["target_xy"], self.target_rest_z + self.INITIAL_FLOOR_CLEARANCE_M] for layout in layouts],
                dtype=torch.float32, device=self.device,
            )
            q = self._yaw_quaternions([layout["target_yaw"] for layout in layouts], self.device)
            self.target_object.set_pose(Pose.create_from_pq(p, q))
            self._reset_actor_velocity(self.target_object, b, self.device)

            for j, (obj, rest_z) in enumerate(zip(self.clutter_objects, self.clutter_rest_z)):
                obj_p = torch.tensor(
                    [[*layout["distractors_xy"][j], rest_z + self.INITIAL_FLOOR_CLEARANCE_M] for layout in layouts],
                    dtype=torch.float32, device=self.device,
                )
                obj_q = self._yaw_quaternions(
                    [layout["distractors_yaw"][j] for layout in layouts], self.device,
                )
                obj.set_pose(Pose.create_from_pq(obj_p, obj_q))
                self._reset_actor_velocity(obj, b, self.device)

            goal_p = torch.tensor(
                [[*layout["goal_xy"], 1e-3] for layout in layouts],
                dtype=torch.float32, device=self.device,
            )
            marker_q = torch.tensor(
                euler2quat(0, np.pi / 2, 0), dtype=torch.float32, device=self.device,
            ).expand(b, -1)
            self.goal_site.set_pose(Pose.create_from_pq(goal_p, marker_q))

    @staticmethod
    def _quaternion_rotation_rad(q0, q1):
        """Shortest orientation distance, batch of (w,x,y,z) quaternions."""
        q0 = q0 / np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-12)
        q1 = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-12)
        return 2.0 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, axis=-1)), 0, 1))

    def _distractor_pose_snapshot(self):
        """Return [N,7] numpy poses; this runner uses a single environment."""
        return self.get_distractor_poses_for_logging().detach().cpu().numpy()[0].copy()

    def reset(self, seed=None, options=None):
        """Let dynamic YCB objects settle BEFORE returning an episode to the monitor.

        _step_action(None) runs physics and robot hold control without counting
        user actions / episode steps. The recorder and DisturbanceMonitor wrap
        env.reset(), so these pre-roll steps are never written to the dataset.
        No velocity is artificially zeroed after settling.
        """
        obs, info = super().reset(seed=seed, options=options)
        if options is not None and "reset_to_env_states" in options:
            return obs, info  # Exact state replay must not be altered.
        if self.settle_steps == 0 and self.idle_verify_steps == 0:
            return obs, info
        if self.num_envs != 1:
            raise ValueError("YCB pre-roll settling currently supports num_envs=1 only")

        requested_poses = self._distractor_pose_snapshot()
        for _ in range(self.settle_steps):
            self._step_action(None)
        settled_poses = self._distractor_pose_snapshot()
        for _ in range(self.idle_verify_steps):
            self._step_action(None)
        verified_poses = self._distractor_pose_snapshot()

        delta_xyz = np.linalg.norm(verified_poses[:, :3] - settled_poses[:, :3], axis=-1)
        delta_rot = self._quaternion_rotation_rad(settled_poses[:, 3:], verified_poses[:, 3:])
        stable = bool(
            np.all(delta_xyz <= self.max_idle_translation_m)
            and np.all(np.rad2deg(delta_rot) <= self.max_idle_rotation_deg)
        )
        request_to_ready = np.linalg.norm(
            verified_poses[:, :3] - requested_poses[:, :3], axis=-1
        )
        report = {
            "settle_steps": self.settle_steps,
            "idle_verify_steps": self.idle_verify_steps,
            "idle_translation_m": delta_xyz.tolist(),
            "idle_rotation_deg": np.rad2deg(delta_rot).tolist(),
            "requested_to_ready_translation_m": request_to_ready.tolist(),
            "idle_stable": stable,
            "idle_translation_tolerance_m": self.max_idle_translation_m,
            "idle_rotation_tolerance_deg": self.max_idle_rotation_deg,
            "ready_distractor_poses": verified_poses.tolist(),
        }
        for layout in self._last_layouts.values():
            layout["initial_stability"] = report
        if not stable:
            import warnings
            warnings.warn(
                "YCB scene remains mobile without robot actions after settling. "
                "Do not use this scene for disturbance training; increase "
                "settle_steps or change object/layout. See initial_stability "
                "in episode_metrics.jsonl.", RuntimeWarning, stacklevel=2,
            )

        # Reset-level observations must match the post-settling state, not
        # the original spawn pose or the pre-roll physics state.
        self._elapsed_steps[:] = 0
        new_info = self.get_info()
        new_info["reconfigure"] = info.get("reconfigure", False)
        new_info["initial_stability"] = {"idle_stable": stable}
        new_obs = self.get_obs(new_info)
        self._last_obs = new_obs
        return new_obs, new_info

    def _get_obs_extra(self, info: dict):
        observation = {
            "tcp_pose": self.agent.tcp_pose.raw_pose,
            "goal_pos": self.goal_site.pose.p,
        }
        if "state" in self.obs_mode:
            # Sim-only object-centric observation; these ARE privileged simulator
            # states, intentionally used after dropping the sim2real requirement.
            b = self.target_object.pose.p.shape[0]
            observation.update({
                "is_grasped": self.agent.is_grasping(self.target_object).float().reshape(b, 1),
                "target_pose": self.target_object.pose.raw_pose,
                "target_linear_velocity": self.target_object.get_linear_velocity(),
                "target_angular_velocity": self.target_object.get_angular_velocity(),
                "distractor_poses": torch.stack(
                    [obj.pose.raw_pose for obj in self.clutter_objects], dim=1,
                ).reshape(b, -1),
                "distractor_linear_velocities": torch.stack(
                    [obj.get_linear_velocity() for obj in self.clutter_objects], dim=1,
                ).reshape(b, -1),
                "distractor_angular_velocities": torch.stack(
                    [obj.get_angular_velocity() for obj in self.clutter_objects], dim=1,
                ).reshape(b, -1),
                "distractor_dimensions": torch.tensor(
                    self.clutter_dimensions, dtype=torch.float32, device=self.device,
                ).reshape(1, -1).expand(b, -1),
                "target_dimensions": torch.tensor(
                    self.target_dimensions, dtype=torch.float32, device=self.device,
                ).reshape(1, -1).expand(b, -1),
            })
        return observation

    def get_distractor_poses_for_logging(self):
        """[B,N,7], quaternion order (w,x,y,z). Stable YCB order."""
        return torch.stack(
            [obj.pose.raw_pose for obj in self.clutter_objects], dim=1,
        )

    def get_distractor_sizes_for_logging(self):
        """Characteristic metres: each YCB actor's max collision-box dimension."""
        return list(self.clutter_size_scales)

    def get_scene_layout_for_logging(self):
        """Layout metadata for matching repeated seeds; not a policy input."""
        def enrich(layout):
            return {
                "target_model_id": self.TARGET_MODEL_ID,
                "distractor_model_ids": list(self.DISTRACTOR_MODEL_IDS),
                "distractor_dimensions_m": self.clutter_dimensions,
                "distractor_sizes_m": self.clutter_size_scales,
                "randomize_layout": self.randomize_layout,
                "distractor_x_range": list(self.distractor_x_range),
                "distractor_y_range": list(self.distractor_y_range),
                **layout,
            }
        if len(self._last_layouts) == 1:
            return enrich(next(iter(self._last_layouts.values())))
        return {env_id: enrich(layout) for env_id, layout in self._last_layouts.items()}

    def evaluate(self):
        target_pos = self.target_object.pose.p
        goal_pos = self.goal_site.pose.p
        xy_error = torch.linalg.norm(target_pos[:, :2] - goal_pos[:, :2], dim=1)
        height_error = torch.abs(target_pos[:, 2] - self.target_rest_z)
        is_obj_placed = (xy_error <= self.goal_thresh) & (
            height_error <= self.GOAL_HEIGHT_TOLERANCE
        )
        is_grasped = self.agent.is_grasping(self.target_object)
        is_robot_static = self.agent.is_static(self.ROBOT_STATIC_THRESHOLD)
        success = is_obj_placed & ~is_grasped & is_robot_static
        return {
            "success": success,
            "is_obj_placed": is_obj_placed,
            "is_grasped": is_grasped,
            "is_released": ~is_grasped,
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_dist = torch.linalg.norm(
            self.target_object.pose.p - self.agent.tcp_pose.p, dim=1,
        )
        reward = 1.0 - torch.tanh(5.0 * tcp_dist)
        is_grasped = info["is_grasped"].float()
        reward += is_grasped
        xy_dist = torch.linalg.norm(
            self.target_object.pose.p[:, :2] - self.goal_site.pose.p[:, :2], dim=1,
        )
        reward += (1.0 - torch.tanh(5.0 * xy_dist)) * is_grasped
        qvel = self.agent.robot.get_qvel()
        static_reward = 1.0 - torch.tanh(5.0 * torch.linalg.norm(qvel, dim=1))
        reward += static_reward * info["is_obj_placed"].float()
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 5.0
