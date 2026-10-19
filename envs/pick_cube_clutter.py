from typing import Any, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, WidowXAI, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

@register_env("PickCubeClutter-v1", max_episode_steps=1000)
class PickCubeClutterEnv(BaseEnv):

    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)
    goal_radius = 0.1

    def __init__(self, *args, robot_uids="xarm6_robotiq", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
                cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
                cfg = PICK_CUBE_CONFIGS["xarm6_robotiq"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.515, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.cube1 = actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size,
                    color=[1, 1, 1, 1],
                    name="cube1",
                    initial_pose=sapien.Pose(p=[0.15, 0, self.cube_half_size]),
                )
        self.cube2 = actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size,
                    color=[0, 1, 0, 1],
                    name="cube2",
                    initial_pose=sapien.Pose(p=[0, 0.15, self.cube_half_size]),
                )
        self.cube3 = actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size,
                    color=[1, 1, 0, 1],
                    name="cube3",
                    initial_pose=sapien.Pose(p=[-0.15, 0, self.cube_half_size]),
                )
        self.cube4 = actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size,
                    color=[1, 0, 1, 1],
                    name="cube4",
                    initial_pose=sapien.Pose(p=[0, -0.15, self.cube_half_size]),
                )
        self.clutter_cubes = [
            self.cube1,
            self.cube2,
            self.cube3,
            self.cube4,
        ]
        self.goal_site = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0.2, 0, 1e-3]),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            self.table_scene.initialize(env_idx)
            self.agent.robot.set_pose(
                sapien.Pose(p=[-0.4, 0, 0])
            )
            # =========================================================
            # 1. Initialize target cube
            # =========================================================
            xyz = torch.zeros((b, 3))
            '''
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            '''
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size

            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
            )

            self.cube.set_pose(
                Pose.create_from_pq(xyz, qs)
            )

            # =========================================================
            # 2. Initialize clutter cubes
            # =========================================================
            clutter_positions = [
                [0.10,  0.00],
                [0.00,  0.10],
                [-0.10, 0.00],
                [0.00, -0.10],
            ]

            for cube, xy in zip(
                self.clutter_cubes,
                clutter_positions
            ):
                clutter_xyz = torch.zeros((b, 3))

                clutter_xyz[:, 0] = xy[0]
                clutter_xyz[:, 1] = xy[1]
                clutter_xyz[:, 2] = self.cube_half_size

                clutter_q = torch.zeros((b, 4))
                clutter_q[:, 0] = 1.0

                cube.set_pose(
                    Pose.create_from_pq(
                        clutter_xyz,
                        clutter_q,
                    )
                )

            # =========================================================
            # 3. Initialize goal
            # =========================================================
            target_region_xyz = xyz + torch.tensor([0.15 + self.goal_radius, 0, 0])
            # set a little bit above 0 so the target is sitting on the table
            target_region_xyz[..., 2] = 1e-3
            self.goal_site.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )

        if "state" in self.obs_mode:

            clutter_poses = torch.stack(
                [
                    cube.pose.raw_pose
                    for cube in self.clutter_cubes
                ],
                dim=1,
            )

            # [B, 4, 7] -> [B, 28]
            clutter_poses = clutter_poses.reshape(
                clutter_poses.shape[0],
                -1,
            )

            clutter_positions = torch.stack(
                [
                    cube.pose.p
                    for cube in self.clutter_cubes
                ],
                dim=1,
            )

            # Calculate TCP -> clutter displacement
            tcp_to_clutter = (
                clutter_positions
                - self.agent.tcp_pose.p[:, None, :]
            )

            # [B, 4, 3] -> [B, 12]
            tcp_to_clutter = tcp_to_clutter.reshape(
                tcp_to_clutter.shape[0],
                -1,
            )

            obs.update(
                obj_pose=self.cube.pose.raw_pose,

                tcp_to_obj_pos=(
                    self.cube.pose.p
                    - self.agent.tcp_pose.p
                ),

                obj_to_goal_pos=(
                    self.goal_site.pose.p
                    - self.cube.pose.p
                ),

                clutter_poses=clutter_poses,

                tcp_to_clutter_pos=tcp_to_clutter,
            )

        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
