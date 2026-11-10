"""Reusable xArm6 pick-and-place environment. Team code defines a SceneSpec.

No dataset files or metric calculations are written in this module.
"""
from __future__ import annotations
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from mani_skill.agents.robots import XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from candid.sim.actors import build_actor, describe_actor
from candid.sim.layout import sample_layout
from candid.sim.distractor_selection import select_distractors
from candid.sim.settling import SettlingConfig, settle_scene


class CANDIDTabletopEnv(BaseEnv):
    """Single-env CPU/MPlib collection reference. Do not use GPU-vectorised
    collection until per-env settling and pose sampling have been implemented.
    """
    SUPPORTED_ROBOTS = ["xarm6_robotiq"]
    agent: XArm6Robotiq
    SCENE = None  # Set to a SceneSpec in a thin envs/*.py module.

    def __init__(self, *args, robot_uids="xarm6_robotiq", randomize_layout=True,
                 num_distractors=None, distractor_seed=0,
                 robot_init_qpos_noise=0.0, max_settle_steps=180,
                 stable_steps=15, linear_vel_threshold=0.005,
                 angular_vel_threshold=0.05,
                 settle_min_steps=20,
                 settle_translation_m=0.0007, settle_rotation_deg=1.0,
                 idle_verify_steps=30,
                 idle_translation_m=0.0010, idle_rotation_deg=1.5,
                 max_initial_xy_shift_mm=10.0,
                 max_initial_tilt_deg=15.0,
                 skip_constructor_settling=False,
                 **kwargs):
        if self.SCENE is None:
            raise ValueError("Subclass must declare SCENE: SceneSpec")
        # Restrict the physical actor set BEFORE BaseEnv builds the scene.
        # Assign to the instance, not to the shared class SceneSpec.
        self.SCENE = select_distractors(
            self.SCENE, count=num_distractors, seed=distractor_seed
        )
        if robot_uids != "xarm6_robotiq":
            raise ValueError("CANDIDTabletopEnv requires xarm6_robotiq")
        if robot_init_qpos_noise != 0:
            raise ValueError("CANDID v1 fixes initial robot qpos noise to zero")
        self.randomize_layout = bool(randomize_layout)
        self.robot_init_qpos_noise = 0.0
        # Retain the old speed thresholds as diagnostics only: instantaneous
        # PhysX contact solver spikes should not reset the settling window.
        self.linear_vel_threshold = float(linear_vel_threshold)
        self.angular_vel_threshold = float(angular_vel_threshold)
        self.settling_config = SettlingConfig(
            max_steps=int(max_settle_steps), min_steps=int(settle_min_steps),
            window_steps=int(stable_steps),
            window_translation_m=float(settle_translation_m),
            window_rotation_deg=float(settle_rotation_deg),
            verify_steps=int(idle_verify_steps),
            verify_translation_m=float(idle_translation_m),
            verify_rotation_deg=float(idle_rotation_deg),
        )
        self._last_layouts = {}
        self._initial_stability = {}
        self.goal_thresh = self.SCENE.goal_xy_tolerance
        self.goal_radius = self.SCENE.goal_marker_radius
        self.max_initial_xy_shift_mm = float(max_initial_xy_shift_mm)
        self.max_initial_tilt_deg = float(max_initial_tilt_deg)

        if not np.isfinite(self.max_initial_xy_shift_mm) or self.max_initial_xy_shift_mm <= 0:
            raise ValueError("max_initial_xy_shift_mm must be positive")

        if not np.isfinite(self.max_initial_tilt_deg) or not 0 < self.max_initial_tilt_deg <= 180:
            raise ValueError("max_initial_tilt_deg must be in (0, 180]")
        # ManiSkill BaseEnv.__init__ performs an implicit seed-2022 reset.
        # Evaluation-suite creation later performs an EXPLICIT seeded reset
        # and fully validates settling. An unlucky bootstrap layout must not
        # prevent that rejection-sampling loop from starting.
        self._skip_constructor_settling = bool(skip_constructor_settling)
        self._in_constructor = True
        try:
            super().__init__(*args, robot_uids=robot_uids, **kwargs)
        finally:
            self._in_constructor = False

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=self.SCENE.camera_eye,
                                   target=self.SCENE.camera_look_at)
        return [CameraConfig("base_camera", pose, 256, 256, np.pi/2, 0.01, 10.0)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=self.SCENE.human_camera_eye,
                                   target=self.SCENE.human_camera_look_at)
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 10.0)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=self.SCENE.robot_base))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.0)
        self.table_scene.build()
        self.target_actor = build_actor(self.scene, self.SCENE.target)
        self.distractor_actors = [build_actor(self.scene, spec)
                                  for spec in self.SCENE.distractors]
        # Compatibility with current xArm6 motion-planning implementation.
        self.cube = self.target_actor
        self.clutter_cubes = self.distractor_actors
        self.goal_site = actors.build_red_white_target(
            self.scene, radius=self.SCENE.goal_marker_radius, thickness=1e-5,
            name="goal_region", body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[*self.SCENE.goal_fixed_xy, 0.001]))

    def _after_reconfigure(self, options: dict):
        self.target_metadata = describe_actor(self.SCENE.target, self.target_actor)
        self.distractor_metadata = [describe_actor(s, a)
                                    for s,a in zip(self.SCENE.distractors, self.distractor_actors)]
        # The legacy cube planner uses this to calculate placement height.
        # It does NOT guarantee a successful grasp on irregular YCB targets.
        self.cube_half_size = self.target_metadata.rest_z

    @staticmethod
    def _yaw_quat(yaws, device):
        a = torch.as_tensor(yaws, dtype=torch.float32, device=device)/2
        return torch.stack([torch.cos(a), torch.zeros_like(a),
                            torch.zeros_like(a), torch.sin(a)], dim=-1)

    @staticmethod
    def _place_actor(obj, layouts, key, idx, z, device):
        if idx is None:
            positions = [layout[key] for layout in layouts]
            yaws = [layout["target_yaw"] for layout in layouts]
        else:
            positions = [layout["distractors_xy"][idx] for layout in layouts]
            yaws = [layout["distractors_yaw"][idx] for layout in layouts]
        xyz = torch.tensor([[*xy, z] for xy in positions],
                           dtype=torch.float32, device=device)
        obj.set_pose(Pose.create_from_pq(xyz, CANDIDTabletopEnv._yaw_quat(yaws, device)))
        zeros = torch.zeros((len(layouts), 3), dtype=torch.float32, device=device)
        obj.set_linear_velocity(zeros)
        obj.set_angular_velocity(zeros)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.agent.robot.set_pose(sapien.Pose(p=self.SCENE.robot_base))
            layouts = []
            for idx in env_idx.detach().cpu().tolist():
                layout = sample_layout(self.SCENE,
                    [self.target_metadata.footprint_radius] +
                    [m.footprint_radius for m in self.distractor_metadata],
                    self._batched_episode_rng[int(idx)], self.randomize_layout)
                self._last_layouts[int(idx)] = layout
                layouts.append(layout)
            self._place_actor(self.target_actor, layouts, "target_xy", None,
                              self.target_metadata.rest_z + 0.001, self.device)
            for i,(obj,meta) in enumerate(zip(self.distractor_actors, self.distractor_metadata)):
                self._place_actor(obj, layouts, None, i, meta.rest_z + 0.001, self.device)
            goal_xyz = torch.tensor([[*l["goal_xy"], .001] for l in layouts],
                                    dtype=torch.float32, device=self.device)
            goal_quat = torch.tensor(euler2quat(0,np.pi/2,0),
                                     dtype=torch.float32, device=self.device).expand(b,-1)
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz, goal_quat))

    def _pose_snapshot_for_settling(self):
        """[N,7] world-frame poses for the current single CPU environment."""
        objects = [self.target_actor, *self.distractor_actors]
        return np.stack([
            obj.pose.raw_pose.detach().cpu().numpy()[0].copy()
            for obj in objects
        ], axis=0)

    
    
    def reset(self, seed=None, options=None):
        obs, original_info = super().reset(seed=seed, options=options)

        if options and "reset_to_env_states" in options:
            return obs, original_info

        if self._in_constructor and self._skip_constructor_settling:
            # Only ManiSkill's throwaway bootstrap reset is skipped. All
            # explicitly requested random candidate resets still run the
            # complete settling/pose-displacement gates below.
            return obs, original_info

        if self.num_envs != 1:
            raise ValueError("CANDID v1 settling supports only num_envs=1")

        if self.control_mode == "pd_joint_pos":
            qpos = self.agent.robot.get_qpos()[0].detach().clone()
            hold_action = self.agent.controller.from_qpos(qpos).unsqueeze(0)
            self.agent.set_action(hold_action)

        specs = [self.SCENE.target, *self.SCENE.distractors]
        names = [spec.name for spec in specs]
        is_sphere = np.array([spec.kind == "sphere" for spec in specs])

        initial = self._pose_snapshot_for_settling()
        peak_xy = np.zeros(len(names), dtype=np.float64)
        peak_tilt = np.zeros(len(names), dtype=np.float64)

        def up_axes(poses):
            q = poses[:, 3:].astype(np.float64)
            q = q / np.maximum(
                np.linalg.norm(q, axis=1, keepdims=True), 1e-12
            )

            w, x, y, z = q.T

            return np.stack([
                2 * (x * z + w * y),
                2 * (y * z - w * x),
                1 - 2 * (x * x + y * y)
            ], axis=1)

        initial_up = up_axes(initial)

        def pose_changes(current):
            xy_shift = np.linalg.norm(
                current[:, :2] - initial[:, :2], axis=1
            )

            current_up = up_axes(current)

            cosine = np.sum(
                initial_up * current_up, axis=1
            )

            tilt = np.degrees(
                np.arccos(np.clip(cosine, -1.0, 1.0))
            )

            tilt[is_sphere] = 0.0

            return xy_shift, tilt

        def settling_step():
            self._step_action(None)

            current = self._pose_snapshot_for_settling()
            xy_shift, tilt = pose_changes(current)

            np.maximum(peak_xy, xy_shift, out=peak_xy)
            np.maximum(peak_tilt, tilt, out=peak_tilt)

        self._initial_stability = settle_scene(
            step=settling_step,
            snapshot=self._pose_snapshot_for_settling,
            config=self.settling_config,
            object_names=names,
        )

        final = self._pose_snapshot_for_settling()
        final_xy, final_tilt = pose_changes(final)

        xy_limit = self.max_initial_xy_shift_mm / 1000.0
        tilt_limit = self.max_initial_tilt_deg

        problems = []
        object_results = {}

        for i, name in enumerate(names):
            object_results[name] = {
                "peak_xy_shift_mm": float(peak_xy[i] * 1000),
                "final_xy_shift_mm": float(final_xy[i] * 1000),
                "peak_tilt_deg": float(peak_tilt[i]),
                "final_tilt_deg": float(final_tilt[i]),
            }

            if peak_xy[i] > xy_limit:
                problems.append(
                    f"{name}: XY shift {peak_xy[i] * 1000:.2f} mm "
                    f"(limit {self.max_initial_xy_shift_mm:.1f} mm)"
                )

            if not is_sphere[i] and peak_tilt[i] > tilt_limit:
                problems.append(
                    f"{name}: tilt {peak_tilt[i]:.2f} deg "
                    f"(limit {tilt_limit:.1f} deg)"
                )

        self._initial_stability["initial_pose_check"] = {
            "passed": len(problems) == 0,
            "max_allowed_xy_shift_mm": self.max_initial_xy_shift_mm,
            "max_allowed_tilt_deg": tilt_limit,
            "max_peak_xy_shift_mm": float(peak_xy.max() * 1000),
            "max_peak_tilt_deg": float(peak_tilt.max()),
            "objects": object_results,
        }

        if problems:
            raise RuntimeError(
                "Invalid initial scene after settling: "
                + "; ".join(problems)
            )

        print(
            "Initial scene validation PASSED: "
            f"max XY shift={peak_xy.max() * 1000:.3f} mm, "
            f"max tilt={peak_tilt.max():.3f} deg",
            flush=True
        )

        self._elapsed_steps[:] = 0

        info = self.get_info()
        info["reconfigure"] = original_info.get("reconfigure", False)

        obs = self.get_obs(info)
        self._last_obs = obs

        return obs, info



    def _get_obs_extra(self, info):
        data = {"tcp_pose": self.agent.tcp_pose.raw_pose,
                "goal_pos": self.goal_site.pose.p}
        if "state" in self.obs_mode:
            B = self.target_actor.pose.p.shape[0]
            m = self.target_metadata
            data.update({
                "target_pose": self.target_actor.pose.raw_pose,
                "target_linear_velocity": self.target_actor.get_linear_velocity(),
                "target_angular_velocity": self.target_actor.get_angular_velocity(),
                "target_dimensions": torch.tensor(m.dimensions, dtype=torch.float32,
                                                   device=self.device).reshape(1,3).expand(B,-1),
                "distractor_poses": torch.stack([o.pose.raw_pose for o in self.distractor_actors], dim=1).reshape(B,-1),
                "distractor_linear_velocities": torch.stack([o.get_linear_velocity() for o in self.distractor_actors],dim=1).reshape(B,-1),
                "distractor_angular_velocities": torch.stack([o.get_angular_velocity() for o in self.distractor_actors],dim=1).reshape(B,-1),
                "distractor_dimensions": torch.tensor([m.dimensions for m in self.distractor_metadata],
                                                      dtype=torch.float32,device=self.device).reshape(1,-1).expand(B,-1),
            })
        return data

    def evaluate(self):
        pos = self.target_actor.pose.p
        goal = self.goal_site.pose.p
        xy_err = torch.linalg.norm(pos[:, :2] - goal[:, :2],dim=1)
        height_err = torch.abs(pos[:, 2] - self.target_metadata.rest_z)
        is_placed = (xy_err <= self.SCENE.goal_xy_tolerance) & (height_err <= self.SCENE.goal_height_tolerance)
        grasped = self.agent.is_grasping(self.target_actor)
        robot_static = self.agent.is_static(0.2)
        return {"success": is_placed & ~grasped & robot_static,
                "is_obj_placed": is_placed, "is_grasped": grasped,
                "is_released": ~grasped, "is_robot_static": robot_static}

    def compute_dense_reward(self, obs, action, info):
        dist = torch.linalg.norm(self.target_actor.pose.p-self.agent.tcp_pose.p,dim=1)
        reward = 1-torch.tanh(5*dist)
        is_grasped = info["is_grasped"].float()
        reward += is_grasped
        place_dist = torch.linalg.norm(self.target_actor.pose.p[:,:2]-self.goal_site.pose.p[:,:2],dim=1)
        reward += (1-torch.tanh(5*place_dist))*is_grasped
        reward += (1-torch.tanh(5*torch.linalg.norm(self.agent.robot.get_qvel(),dim=1)))*info["is_obj_placed"].float()
        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs,action,info)/5

    # External data components may inspect this registry without requiring
    # each scene author to implement logging methods manually.
    def get_scene_layout_for_logging(self):
        layout = next(iter(self._last_layouts.values())).copy()
        layout["initial_stability"] = self._initial_stability
        return layout
