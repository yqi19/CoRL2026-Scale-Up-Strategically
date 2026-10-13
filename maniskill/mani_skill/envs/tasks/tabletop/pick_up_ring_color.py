"""Pick a ring: grasp and place to goal. Same workspace as Cube/Sphere: x,y in [-0.1, 0.1]."""
from typing import Any, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import (
    OBJECT_COLORS_VERB,
    PICK_CUBE_CONFIGS,
    apply_object_sizes_to_cfg,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PickRingColor-v1", max_episode_steps=50)
class PickRingColorEnv(BaseEnv):
    """Pick a ring and place at goal. Workspace: x,y in [-0.1, 0.1] m."""

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]
    goal_thresh = 0.025
    pick_height_thresh = 0.08  # min z lift for success (same scale as push/pull y_delta_thresh)

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, object_color_id=0, show_goal_site=False, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_color_id = object_color_id  # 0=red, 1=yellow, 2=blue
        self.show_goal_site = show_goal_site  # if False, hide the green goal sphere in render (default no viz)
        self.object_color = OBJECT_COLORS_VERB[object_color_id]
        cfg = dict(PICK_CUBE_CONFIGS.get(robot_uids, PICK_CUBE_CONFIGS["panda_wristcam"]))
        apply_object_sizes_to_cfg(cfg, kwargs.pop("object_sizes_path", None), kwargs.pop("object_sizes", None))
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.ring_radius = cfg.get("ring_radius", 0.03)
        self.ring_inner_radius = cfg.get("ring_inner_radius", 0.015)
        self.ring_half_height = cfg.get("ring_half_height", 0.008)
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
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.obj = actors.build_ring(
            self.scene,
            radius=self.ring_radius,
            half_height=self.ring_half_height,
            color=self.object_color,
            name="ring",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.ring_half_height]),
            inner_radius=self.ring_inner_radius,
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        if self.show_goal_site:
            self._hidden_objects.append(self.goal_site)
        else:
            self.goal_site.hide_visual()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            if options and "obj_xy" in options:
                obj_xy = torch.tensor(options["obj_xy"], dtype=torch.float32, device=self.device)
                xyz[:, 0] = obj_xy[0]
                xyz[:, 1] = obj_xy[1]
            else:
                xyz[:, :2] = (
                    torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                    - self.cube_spawn_half_size
                )
                xyz[:, 0] += self.cube_spawn_center[0]
                xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.ring_half_height
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(xyz, qs))
            self._obj_init_z = xyz[:, 2].clone()

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,
            )
        return obs

    def evaluate(self):
        obj_z = self.obj.pose.p[..., 2]
        is_lifted = (obj_z - self._obj_init_z) >= self.pick_height_thresh
        is_robot_static = self.agent.is_static(0.2)
        is_grasped = self.agent.is_grasping(self.obj)
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.obj.pose.p, axis=1)
            <= self.goal_thresh
        )
        return {
            "success": is_lifted & is_robot_static,
            "is_lifted": is_lifted,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            "is_obj_placed": is_obj_placed,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward
        is_grasped = info["is_grasped"]
        reward += is_grasped
        obj_z = self.obj.pose.p[..., 2]
        lift_height = (obj_z - self._obj_init_z).clamp(min=0)
        lift_reward = 1 - torch.tanh(5 * (self.pick_height_thresh - lift_height).clamp(min=0))
        reward += lift_reward * is_grasped
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_lifted"]
        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
