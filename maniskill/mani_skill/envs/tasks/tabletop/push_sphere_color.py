"""Push a sphere: success = x increases (push forward along x-axis), y roughly same. No target drawn."""
from typing import Any, Union

import numpy as np
import sapien
import torch

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
from mani_skill.utils.structs.types import Array


@register_env("PushSphereColor-v1", max_episode_steps=50)
class PushSphereColorEnv(BaseEnv):
    """
    Push a sphere forward: success when x increases and y stays roughly the same. No target drawn.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PushCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]
    x_delta_thresh = 0.08   # min x increase (push forward)
    y_tolerance = 0.05

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, object_color_id=0, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_color_id = object_color_id  # 0=red, 1=yellow, 2=blue
        self.object_color = OBJECT_COLORS_VERB[object_color_id]
        cfg = dict(PICK_CUBE_CONFIGS.get(robot_uids, PICK_CUBE_CONFIGS["panda_wristcam"]))
        apply_object_sizes_to_cfg(cfg, kwargs.pop("object_sizes_path", None), kwargs.pop("object_sizes", None))
        self.radius = cfg["radius"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
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
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.obj = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=self.object_color,
            name="sphere",
            body_type="dynamic",
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            if options and "obj_xy" in options:
                obj_xy = torch.tensor(options["obj_xy"], dtype=torch.float32, device=self.device)
                xyz[..., 0] = obj_xy[0]
                xyz[..., 1] = obj_xy[1]
            else:
                xyz[..., :2] = (
                    torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                    - self.cube_spawn_half_size
                )
                xyz[..., 0] += self.cube_spawn_center[0]
                xyz[..., 1] += self.cube_spawn_center[1]
            xyz[..., 2] = self.radius
            q = [1, 0, 0, 0]

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)
            self._obj_init_xy = xyz[..., :2].clone()

    def evaluate(self):
        obj_xy = self.obj.pose.p[..., :2]
        x_increased = (obj_xy[..., 0] - self._obj_init_xy[..., 0]) >= self.x_delta_thresh
        y_same = torch.abs(obj_xy[..., 1] - self._obj_init_xy[..., 1]) < self.y_tolerance
        on_table = self.obj.pose.p[..., 2] < self.radius + 5e-3
        is_success = x_increased & y_same & on_table
        return {"success": is_success}

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                obj_init_xy=self._obj_init_xy,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tcp_push_pose = self.obj.pose.p + torch.tensor(
            [-self.radius - 0.005, 0, 0], device=self.device
        )
        tcp_to_push_pose = tcp_push_pose - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = reaching_reward

        reached = tcp_to_push_pose_dist < 0.01
        obj_xy = self.obj.pose.p[..., :2]
        x_delta = obj_xy[..., 0] - self._obj_init_xy[..., 0]
        y_dev = torch.abs(obj_xy[..., 1] - self._obj_init_xy[..., 1])
        place_reward = (1 - torch.tanh(5 * (-x_delta))) * (1 - torch.tanh(5 * y_dev))
        reward += place_reward * reached

        desired_obj_z = self.radius
        current_obj_z = self.obj.pose.p[..., 2]
        z_deviation = torch.abs(current_obj_z - desired_obj_z)
        z_reward = 1 - torch.tanh(5 * z_deviation)
        reward += place_reward * z_reward * reached

        reward[info["success"]] = 4
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0
