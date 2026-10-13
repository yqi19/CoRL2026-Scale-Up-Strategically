"""
Unified tabletop task: verb × object shape × color, optional distractors (0–3).

Verbs (success semantics):
  lift   — **In world frame:** object center ``z > lift_min_world_z`` (default 0) **and**
           **relative to episode start:** ``z - z_init >= pick_height_thresh`` (scaled by difficulty).
           Robot nearly static. Aliases: pick, pick_up, pickup.
  push   — +x displacement, |Δy| small, stay on table.
  pull   — −x displacement, |Δy| small, stay on table.
  rotate — **eval:** |Δyaw| > ``rotate_eval_min_abs_delta_rad`` (default 50°, scaled by difficulty).
           MP demo: **small** vertical lift then wrist **90°** about world z (not a full ``lift`` height).
  grasp  — **In-place:** planar slip ``‖xy - xy_init‖`` and ``|z - z_init|`` below small caps;
           gripper **sufficiently closed** (last two qpos sum below a loose cap — allows partial closure);
           **finger–object contact:** sum of contact-force norms on both fingers > threshold; object still on table.
           Alias: press.
  slide  — |Δy| large (motion along y), |Δx| small, stay on table.
  nudge  — like push but smaller required +x slip (``nudge_delta_thresh``, scaled).
  touch  — gripper closed, TCP near object, finger–object contact only (no table-force requirement).

``task_difficulty`` (default 1.0): >1 increases required displacements / pick height / eval rotation / jiggle, tightens spawn spacing.

Distractors: up to 3 extra objects. Per slot, the (shape, color_id) comes from ``distractor_specs[i]``
if given; else from the target if ``distractor_match_target_geometry[i]`` is True (same look,
different scale, used for relative-size tasks); else it is sampled uniformly at random and
resampled if it exactly duplicates the target.
Per-episode ``reconfiguration_freq=1`` by default.

OOD / factor-bias **evaluation only** (optional): pass ``ood_dual_eval`` at ``gym.make`` or per
``reset(..., options={"ood_dual_eval": {...}})``. When set, each ``evaluate()`` also fills
``success_first_axis`` and ``success_second_axis`` (same shape as ``success``) via
``compute_ood_dual_success`` — the motion planner and task ``verb`` are unchanged; this is
extra success masks for logging / metrics only.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import (
    OBJECT_COLORS_VERB,
    PICK_CUBE_CONFIGS,
)
from mani_skill.envs.tasks.tabletop.tabletop_composite_objects import (
    build_tabletop_object,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

VERB_CHOICES = ("lift", "push", "pull", "rotate", "grasp", "slide", "nudge", "touch")
SHAPE_CHOICES = ("cube", "sphere", "cup", "ring", "car", "pyramid", "star")


def _normalize_object_shape(shape: str) -> str:
    s = shape.lower().strip()
    if s in ("can", "mug", "cylinder"):
        return "cup"
    if s == "bottle":
        raise ValueError(
            f"object_shape 'bottle' is not supported; use one of {SHAPE_CHOICES}"
        )
    return s


def _normalize_verb(verb: str) -> str:
    v = verb.lower().replace(" ", "_").replace("-", "_")
    if v in ("pick_up", "pickup", "pick"):
        return "lift"
    if v == "press":
        return "grasp"
    if v not in VERB_CHOICES:
        raise ValueError(
            f"verb must be one of {VERB_CHOICES} or pick/pick_up/press, got {verb!r}"
        )
    return v


def _quat_yaw_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Yaw (about world z) from wxyz quaternion, shape (..., 4) -> (...)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _angle_wrap_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


@register_env("VerbObjectColor-v1", max_episode_steps=120)
class VerbObjectColorEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]

    # thresholds (meters / radians)
    # Object Δz vs episode start for ``lift`` / pick_up success (scaled by ``task_difficulty``).
    pick_height_thresh = 0.08
    goal_thresh = 0.028
    # slide: min |Δy|; push/pull use ``push_pull_delta_thresh`` only
    xy_delta_thresh = 0.08
    push_pull_delta_thresh = 0.055
    y_tolerance = 0.05
    x_tolerance_slide = 0.05
    # rotate: evaluation uses absolute yaw change from episode start (see evaluate).
    rotate_eval_min_abs_delta_rad = np.deg2rad(50.0)
    on_table_eps = 0.012
    # lift: world-z floor (m) plus relative rise from episode init (see evaluate).
    lift_min_world_z = 0.0
    # grasp: object stays near spawn; gripper not wide open; real finger–object contact.
    grasp_xy_slip_max = 0.018
    grasp_z_slip_max = 0.010
    grasp_gripper_qpos_sum_max = 0.068
    grasp_min_contact_force = 0.10
    nudge_delta_thresh = 0.032
    touch_tcp_xy_tol = 0.055
    touch_tcp_z_above = 0.06
    touch_tcp_z_below = -0.025
    touch_min_pad_force = 0.14
    touch_lift_max = 0.028

    max_goal_height = 0.3
    # Spawn pose: small +z above nominal rest to reduce table penetration; xy clearance between all bodies.
    spawn_z_lift = 0.004
    spawn_xy_min_sep = 0.115
    spawn_placement_max_attempts = 120

    def __init__(
        self,
        *args,
        robot_uids: str = "panda_wristcam",
        robot_init_qpos_noise: float = 0.02,
        verb: str = "lift",
        object_shape: str = "cube",
        object_color_id: int = 0,
        distractor_max: int = 3,
        object_size_jiggle: float = 0.2,
        target_size_scale: float | None = None,
        distractor_size_scales: list[float] | tuple[float, ...] | None = None,
        task_difficulty: float = 1.0,
        show_goal_site: bool = False,
        reconfiguration_freq: Optional[int] = None,
        distractor_specs: Optional[list[Optional[tuple[str, int]]]] = None,
        distractor_match_target_geometry: tuple[bool, ...] | None = None,
        ood_dual_eval: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        # Distractor shape/color are sampled each reset in _load_scene → need rebuild every episode.
        if reconfiguration_freq is None:
            reconfiguration_freq = 1
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.verb = _normalize_verb(verb)
        self.object_shape = _normalize_object_shape(object_shape)
        if self.object_shape not in SHAPE_CHOICES:
            raise ValueError(f"object_shape must be one of {SHAPE_CHOICES}")
        self.object_color_id = int(object_color_id)
        if not 0 <= self.object_color_id < len(OBJECT_COLORS_VERB):
            raise ValueError(
                f"object_color_id must be in [0, {len(OBJECT_COLORS_VERB)}), got {object_color_id}"
            )
        self.distractor_max = max(0, min(3, int(distractor_max)))
        self.object_size_jiggle = max(0.0, float(object_size_jiggle))
        self.target_size_scale = (
            None if target_size_scale is None else max(0.2, float(target_size_scale))
        )
        self.distractor_size_scales = (
            None
            if distractor_size_scales is None
            else [max(0.2, float(x)) for x in distractor_size_scales]
        )
        self.task_difficulty = float(np.clip(float(task_difficulty), 0.5, 3.0))
        self.show_goal_site = show_goal_site
        self.object_color = OBJECT_COLORS_VERB[self.object_color_id]
        if distractor_specs is not None and len(distractor_specs) != 3:
            raise ValueError(
                "distractor_specs must be None or a length-3 list "
                "(tuple[str,int] = (shape, color_id) per slot, or None for random)"
            )
        self._distractor_specs = distractor_specs
        if distractor_match_target_geometry is None:
            self.distractor_match_target_geometry = (False, False, False)
        else:
            g = tuple(bool(x) for x in distractor_match_target_geometry)
            if len(g) != 3:
                raise ValueError("distractor_match_target_geometry must have length 3")
            self.distractor_match_target_geometry = g
        self._ood_dual_eval_default = ood_dual_eval

        dm = max(0.0, self.task_difficulty - 1.0)
        self._eff_pick_height_thresh = self.pick_height_thresh * (1.0 + 0.12 * dm)
        self._eff_push_pull_delta = self.push_pull_delta_thresh * (1.0 + 0.10 * dm)
        self._eff_slide_xy_delta = self.xy_delta_thresh * (1.0 + 0.08 * dm)
        self._eff_nudge_delta = self.nudge_delta_thresh * (1.0 + 0.10 * dm)
        self._eff_spawn_sep = max(0.088, float(self.spawn_xy_min_sep) * (1.0 - 0.05 * dm))
        self._eff_object_size_jiggle = self.object_size_jiggle * (1.0 + 0.25 * dm)
        self._eff_rotate_min_rad = float(self.rotate_eval_min_abs_delta_rad) * (
            1.0 + 0.10 * dm
        )
        self._eff_grasp_xy_slip = float(self.grasp_xy_slip_max) * (1.0 - 0.06 * dm)
        self._eff_grasp_z_slip = float(self.grasp_z_slip_max) * (1.0 - 0.06 * dm)
        self._eff_grasp_contact = float(self.grasp_min_contact_force) * (1.0 + 0.10 * dm)
        self._eff_touch_pad = self.touch_min_pad_force * (1.0 + 0.15 * dm)

        cfg = dict(PICK_CUBE_CONFIGS.get(robot_uids, PICK_CUBE_CONFIGS["panda_wristcam"]))
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        self._obj_rest_z = 0.02
        self._distractor_rest_z: list[float] = []
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

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

    def _sample_tabletop_size_scale(self) -> float:
        """Uniform scale in [1-jiggle, 1+jiggle]; uses env 0's stream (shared geometry across sub-scenes)."""
        j = self._eff_object_size_jiggle
        if j <= 0:
            return 1.0
        lo, hi = 1.0 - j, 1.0 + j
        raw = self._episode_rng.uniform(lo, hi)
        return float(np.asarray(raw).reshape(-1)[0])

    def _sample_distractor_shape_color_id(self) -> tuple[str, int]:
        """Uniform random (shape, color_id); resample if both match the target."""
        rng = self._episode_rng
        n_colors = len(OBJECT_COLORS_VERB)
        shapes = list(SHAPE_CHOICES)
        for _ in range(256):
            sh = str(rng.choice(shapes))
            cid = int(rng.randint(n_colors))
            if sh != self.object_shape or cid != self.object_color_id:
                return sh, cid
        # Degenerate case (e.g. single color): force a non-duplicate pair
        for sh in shapes:
            if sh != self.object_shape:
                return sh, self.object_color_id
        return self.object_shape, (self.object_color_id + 1) % n_colors

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.obj, self._obj_rest_z = build_tabletop_object(
            self.scene,
            self.object_shape,
            self.object_color,
            name="target_obj",
            size_scale=(
                self.target_size_scale
                if self.target_size_scale is not None
                else self._sample_tabletop_size_scale()
            ),
        )
        # Alias for motion-planning code that expects ``cube``
        self.cube = self.obj

        self.goal_site = None
        if self.verb == "lift":
            from mani_skill.utils.building import actors

            self.goal_site = actors.build_sphere(
                self.scene,
                radius=self.goal_thresh,
                color=[0, 1, 0, 1],
                name="goal_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            # GPU sim: hide_visual() reads cuda_rigid_body_data via .pose, not valid during _load_scene.
            self._goal_site_hide_after_episode_init = False
            if not self.show_goal_site:
                if self.scene.gpu_sim_enabled:
                    self._goal_site_hide_after_episode_init = True
                else:
                    self.goal_site.hide_visual()

        self.distractor_actors = []
        self._distractor_rest_z = []
        # Three slots: explicit spec > match target geometry > uniform random (reject exact copy of target).
        for i in range(3):
            if (
                self._distractor_specs is not None
                and self._distractor_specs[i] is not None
            ):
                sh, cid = self._distractor_specs[i]
                sh = _normalize_object_shape(str(sh))
                cid = int(cid)
                if not 0 <= cid < len(OBJECT_COLORS_VERB):
                    raise ValueError(f"invalid distractor color_id {cid}")
                if sh not in SHAPE_CHOICES:
                    raise ValueError(f"invalid distractor shape {sh!r}")
            elif self.distractor_match_target_geometry[i]:
                sh, cid = self.object_shape, self.object_color_id
            else:
                sh, cid = self._sample_distractor_shape_color_id()
            col = OBJECT_COLORS_VERB[cid]
            if self.distractor_size_scales is not None and i < len(self.distractor_size_scales):
                d_scale = self.distractor_size_scales[i]
            else:
                d_scale = self._sample_tabletop_size_scale()
            act, rz = build_tabletop_object(
                self.scene,
                sh,
                col,
                name=f"distractor_{i}",
                size_scale=d_scale,
            )
            self.distractor_actors.append(act)
            self._distractor_rest_z.append(rz)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if options and "ood_dual_eval" in options:
                self._ood_dual_eval = options["ood_dual_eval"]
            else:
                self._ood_dual_eval = self._ood_dual_eval_default

            # Target pose (same workspace as other verb×color envs)
            xyz = torch.zeros((b, 3))
            if options and "obj_xy" in options:
                oxy = torch.tensor(options["obj_xy"], dtype=torch.float32, device=self.device)
                xyz[:, 0] = oxy[0]
                xyz[:, 1] = oxy[1]
            else:
                xyz[:, :2] = (
                    torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                    - self.cube_spawn_half_size
                )
                xyz[:, 0] += self.cube_spawn_center[0]
                xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self._obj_rest_z + self.spawn_z_lift
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self._obj_init_xy = xyz[:, :2].clone()
            self._obj_init_z = xyz[:, 2].clone()
            self._obj_init_yaw = _quat_yaw_wxyz(self.obj.pose.q)

            if self.goal_site is not None:
                goal_xyz = torch.zeros((b, 3), device=self.device)
                goal_xyz[:, :2] = (
                    torch.rand((b, 2), device=self.device) * self.cube_spawn_half_size * 2
                    - self.cube_spawn_half_size
                )
                goal_xyz[:, 0] += self.cube_spawn_center[0]
                goal_xyz[:, 1] += self.cube_spawn_center[1]
                goal_xyz[:, 2] = torch.rand((b,), device=self.device) * self.max_goal_height + xyz[:, 2]
                gq = torch.zeros((b, 4), device=self.device)
                gq[:, 0] = 1.0
                self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz, q=gq))
                if getattr(self, "_goal_site_hide_after_episode_init", False):
                    self.goal_site.hide_visual()
                    self._goal_site_hide_after_episode_init = False

            # Distractor count 0..distractor_max (same for all parallel envs this step)
            if options and "num_distractors" in options:
                n_d = int(options["num_distractors"])
            else:
                n_d = int(torch.randint(0, self.distractor_max + 1, (1,), device=self.device).item())
            n_d = max(0, min(3, n_d))
            self._active_distractors = n_d

            self._distractor_init_xy = [None, None, None]
            self._distractor_init_z = [None, None, None]
            self._distractor_init_yaw = [None, None, None]

            hq = torch.zeros((b, 4), device=self.device)
            hq[:, 0] = 1.0
            hide_pose = Pose.create_from_pq(
                p=torch.cat(
                    [
                        torch.zeros((b, 2), device=self.device),
                        torch.full((b, 1), -0.5, device=self.device),
                    ],
                    dim=-1,
                ),
                q=hq,
            )
            # xy positions already placed this episode (target first); each distractor must clear all of them.
            placed_xy: list[torch.Tensor] = [xyz[:, :2].clone()]

            for di, d_actor in enumerate(self.distractor_actors):
                if di >= n_d:
                    d_actor.set_pose(hide_pose)
                    continue
                rz = self._distractor_rest_z[di] + self.spawn_z_lift
                placed = False
                for _ in range(self.spawn_placement_max_attempts):
                    dx = (
                        torch.rand(b, device=self.device)
                        * 2
                        * self.cube_spawn_half_size
                        - self.cube_spawn_half_size
                    )
                    dy = (
                        torch.rand(b, device=self.device)
                        * 2
                        * self.cube_spawn_half_size
                        - self.cube_spawn_half_size
                    )
                    dx = dx + self.cube_spawn_center[0]
                    dy = dy + self.cube_spawn_center[1]
                    cand_xy = torch.stack([dx, dy], dim=-1)
                    ok = torch.ones(b, dtype=torch.bool, device=self.device)
                    for pref in placed_xy:
                        sep = torch.linalg.norm(cand_xy - pref, dim=1)
                        ok &= sep > self._eff_spawn_sep
                    if torch.all(ok):
                        dxyz = torch.zeros((b, 3), device=self.device)
                        dxyz[:, 0] = dx
                        dxyz[:, 1] = dy
                        dxyz[:, 2] = rz
                        d_actor.set_pose(
                            Pose.create_from_pq(
                                p=dxyz,
                                q=randomization.random_quaternions(
                                    b, lock_x=True, lock_y=True
                                ),
                            )
                        )
                        self._distractor_init_xy[di] = cand_xy.clone()
                        self._distractor_init_z[di] = dxyz[:, 2].clone()
                        self._distractor_init_yaw[di] = _quat_yaw_wxyz(
                            d_actor.pose.q
                        ).clone()
                        placed_xy.append(cand_xy.clone())
                        placed = True
                        break
                if not placed:
                    d_actor.set_pose(hide_pose)

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs["obj_pose"] = self.obj.pose.raw_pose
            if self.goal_site is not None:
                obs["goal_pos"] = self.goal_site.pose.p
        return obs

    def compute_success_for_verb_on_actor(
        self,
        verb: str,
        obj_actor,
        init_xy: torch.Tensor,
        init_z: torch.Tensor,
        init_yaw: torch.Tensor,
        rest_z_for_on_table: float,
    ) -> torch.Tensor:
        """Success mask for an alternate verb on an arbitrary rigid body (batched env dim 0)."""
        v = _normalize_verb(verb)
        obj_xy = obj_actor.pose.p[..., :2]
        obj_z = obj_actor.pose.p[..., 2]
        on_table = obj_z < float(rest_z_for_on_table) + self.on_table_eps

        if v == "lift":
            delta_z = obj_z - init_z
            above_world = obj_z > float(self.lift_min_world_z)
            is_lifted = (delta_z >= self._eff_pick_height_thresh) & above_world
            is_robot_static = self.agent.is_static(0.2)
            return is_lifted & is_robot_static
        if v == "push":
            dx = obj_xy[..., 0] - init_xy[..., 0]
            dy = torch.abs(obj_xy[..., 1] - init_xy[..., 1])
            return (dx >= self._eff_push_pull_delta) & (dy < self.y_tolerance) & on_table
        if v == "pull":
            dx = init_xy[..., 0] - obj_xy[..., 0]
            dy = torch.abs(obj_xy[..., 1] - init_xy[..., 1])
            return (dx >= self._eff_push_pull_delta) & (dy < self.y_tolerance) & on_table
        if v == "nudge":
            dx = obj_xy[..., 0] - init_xy[..., 0]
            dy = torch.abs(obj_xy[..., 1] - init_xy[..., 1])
            return (dx >= self._eff_nudge_delta) & (dy < self.y_tolerance) & on_table
        if v == "slide":
            dy_abs = torch.abs(obj_xy[..., 1] - init_xy[..., 1])
            dx_abs = torch.abs(obj_xy[..., 0] - init_xy[..., 0])
            return (
                (dy_abs >= self._eff_slide_xy_delta)
                & (dx_abs < self.x_tolerance_slide)
                & on_table
            )
        if v == "rotate":
            yaw = _quat_yaw_wxyz(obj_actor.pose.q)
            delta = _angle_wrap_diff(yaw, init_yaw)
            delta_abs = torch.abs(delta)
            min_r = float(self._eff_rotate_min_rad)
            return delta_abs > min_r
        if v == "grasp":
            slip_xy = torch.linalg.norm(obj_xy - init_xy, dim=1)
            slip_z = torch.abs(obj_z - init_z)
            in_place = (slip_xy <= self._eff_grasp_xy_slip) & (
                slip_z <= self._eff_grasp_z_slip
            )
            qpos = self.agent.robot.get_qpos()
            finger = qpos[..., -2:].sum(dim=-1)
            gripper_closed_enough = finger < self.grasp_gripper_qpos_sum_max
            f1 = torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger1_link, obj_actor
                ),
                dim=1,
            )
            f2 = torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger2_link, obj_actor
                ),
                dim=1,
            )
            f_contact = f1 + f2
            has_contact = f_contact > self._eff_grasp_contact
            return in_place & gripper_closed_enough & has_contact & on_table
        if v == "touch":
            tcp = self.agent.tcp.pose.p
            dxy = torch.linalg.norm(tcp[..., :2] - obj_actor.pose.p[..., :2], dim=1)
            dz = tcp[..., 2] - obj_actor.pose.p[..., 2]
            lift = torch.abs(obj_z - init_z)
            qpos = self.agent.robot.get_qpos()
            finger = qpos[..., -2:].sum(dim=-1)
            closed = finger < 0.055
            f1 = torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger1_link, obj_actor
                ),
                dim=1,
            )
            f2 = torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger2_link, obj_actor
                ),
                dim=1,
            )
            f_touch = f1 + f2
            pose_ok = (
                (dxy < self.touch_tcp_xy_tol)
                & (dz < self.touch_tcp_z_above)
                & (dz > self.touch_tcp_z_below)
            )
            return (
                closed
                & pose_ok
                & (f_touch > self._eff_touch_pad)
                & (lift < self.touch_lift_max)
                & on_table
            )
        raise RuntimeError(f"Unhandled verb {v!r} in compute_success_for_verb_on_actor")

    def compute_ood_dual_success(
        self,
        *,
        verb_row: str,
        verb_col: str,
        layout: Literal["single_target", "color_object_dual"] = "single_target",
        metric_mode: Literal["verb", "push_only"] = "verb",
    ) -> dict[str, Any]:
        """Two success bits for OOD evaluation (verb / color-shape bias analysis).

        ``verb_row`` / ``verb_col`` are normally ``TRAINING_VERBS[i]`` and ``TRAINING_VERBS[j]`` for
        OOD index pair ``(i, j)``. With ``metric_mode="push_only"``, both checks use **push**
        displacement semantics on the corresponding rigid body.
        """
        if metric_mode == "push_only":
            v0 = v1 = "push"
        else:
            v0 = verb_row
            v1 = verb_col
        if layout == "single_target":
            ok0 = self.compute_success_for_verb_on_actor(
                v0,
                self.obj,
                self._obj_init_xy,
                self._obj_init_z,
                self._obj_init_yaw,
                self._obj_rest_z,
            )
            ok1 = self.compute_success_for_verb_on_actor(
                v1,
                self.obj,
                self._obj_init_xy,
                self._obj_init_z,
                self._obj_init_yaw,
                self._obj_rest_z,
            )
        elif layout == "color_object_dual":
            if len(self.distractor_actors) < 2:
                raise RuntimeError("color_object_dual needs at least two distractor slots")
            if self._distractor_init_xy[0] is None or self._distractor_init_xy[1] is None:
                raise RuntimeError(
                    "missing distractor episode inits; use num_distractors>=2 in reset options"
                )
            d0, d1 = self.distractor_actors[0], self.distractor_actors[1]
            rz0 = self._distractor_rest_z[0]
            rz1 = self._distractor_rest_z[1]
            ok0 = self.compute_success_for_verb_on_actor(
                v0,
                d0,
                self._distractor_init_xy[0],
                self._distractor_init_z[0],
                self._distractor_init_yaw[0],
                rz0,
            )
            ok1 = self.compute_success_for_verb_on_actor(
                v1,
                d1,
                self._distractor_init_xy[1],
                self._distractor_init_z[1],
                self._distractor_init_yaw[1],
                rz1,
            )
        else:
            raise ValueError(f"unknown layout {layout!r}")
        return {
            "success_first_axis": ok0,
            "success_second_axis": ok1,
            "verb_first": _normalize_verb(v0),
            "verb_second": _normalize_verb(v1),
            "layout": layout,
            "metric_mode": metric_mode,
        }

    def evaluate(self):
        obj_xy = self.obj.pose.p[..., :2]
        obj_z = self.obj.pose.p[..., 2]
        on_table = obj_z < self._obj_rest_z + self.on_table_eps

        is_grasped = self.agent.is_grasping(self.obj)
        info_out: dict[str, Any] = {
            "is_grasped": is_grasped,
            "success": torch.zeros(obj_xy.shape[0], dtype=torch.bool, device=self.device),
        }

        if self.verb == "lift":
            delta_z = obj_z - self._obj_init_z
            above_world = obj_z > float(self.lift_min_world_z)
            is_lifted = (delta_z >= self._eff_pick_height_thresh) & above_world
            is_robot_static = self.agent.is_static(0.2)
            ok = is_lifted & is_robot_static
            info_out["success"] = ok
            info_out["is_lifted"] = is_lifted
            info_out["lift_delta_z"] = delta_z
            info_out["lift_above_world_z"] = above_world
        elif self.verb == "push":
            dx = obj_xy[..., 0] - self._obj_init_xy[..., 0]
            dy = torch.abs(obj_xy[..., 1] - self._obj_init_xy[..., 1])
            ok = (dx >= self._eff_push_pull_delta) & (dy < self.y_tolerance) & on_table
            info_out["success"] = ok
        elif self.verb == "pull":
            dx = self._obj_init_xy[..., 0] - obj_xy[..., 0]
            dy = torch.abs(obj_xy[..., 1] - self._obj_init_xy[..., 1])
            ok = (dx >= self._eff_push_pull_delta) & (dy < self.y_tolerance) & on_table
            info_out["success"] = ok
        elif self.verb == "nudge":
            dx = obj_xy[..., 0] - self._obj_init_xy[..., 0]
            dy = torch.abs(obj_xy[..., 1] - self._obj_init_xy[..., 1])
            ok = (dx >= self._eff_nudge_delta) & (dy < self.y_tolerance) & on_table
            info_out["success"] = ok
        elif self.verb == "slide":
            dy_abs = torch.abs(obj_xy[..., 1] - self._obj_init_xy[..., 1])
            dx_abs = torch.abs(obj_xy[..., 0] - self._obj_init_xy[..., 0])
            ok = (
                (dy_abs >= self._eff_slide_xy_delta)
                & (dx_abs < self.x_tolerance_slide)
                & on_table
            )
            info_out["success"] = ok
        elif self.verb == "rotate":
            yaw = _quat_yaw_wxyz(self.obj.pose.q)
            delta = _angle_wrap_diff(yaw, self._obj_init_yaw)
            delta_abs = torch.abs(delta)
            min_r = float(self._eff_rotate_min_rad)
            ok = delta_abs > min_r
            info_out["success"] = ok
            info_out["delta_yaw"] = delta_abs
        elif self.verb == "grasp":
            slip_xy = torch.linalg.norm(obj_xy - self._obj_init_xy, dim=1)
            slip_z = torch.abs(obj_z - self._obj_init_z)
            in_place = (slip_xy <= self._eff_grasp_xy_slip) & (
                slip_z <= self._eff_grasp_z_slip
            )
            qpos = self.agent.robot.get_qpos()
            finger = qpos[..., -2:].sum(dim=-1)
            gripper_closed_enough = finger < self.grasp_gripper_qpos_sum_max
            f1 = torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger1_link, self.obj
                ),
                dim=1,
            )
            f2 = torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger2_link, self.obj
                ),
                dim=1,
            )
            f_contact = f1 + f2
            has_contact = f_contact > self._eff_grasp_contact
            ok = in_place & gripper_closed_enough & has_contact & on_table
            info_out["success"] = ok
            info_out["grasp_slip_xy"] = slip_xy
            info_out["grasp_slip_z"] = slip_z
            info_out["grasp_f_contact"] = f_contact
        elif self.verb == "touch":
            tcp = self.agent.tcp.pose.p
            dxy = torch.linalg.norm(tcp[..., :2] - self.obj.pose.p[..., :2], dim=1)
            dz = tcp[..., 2] - self.obj.pose.p[..., 2]
            lift = torch.abs(obj_z - self._obj_init_z)
            qpos = self.agent.robot.get_qpos()
            finger = qpos[..., -2:].sum(dim=-1)
            closed = finger < 0.055
            f1 = torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger1_link, self.obj
                ),
                dim=1,
            )
            f2 = torch.linalg.norm(
                self.scene.get_pairwise_contact_forces(
                    self.agent.finger2_link, self.obj
                ),
                dim=1,
            )
            f_touch = f1 + f2
            pose_ok = (
                (dxy < self.touch_tcp_xy_tol)
                & (dz < self.touch_tcp_z_above)
                & (dz > self.touch_tcp_z_below)
            )
            ok = (
                closed
                & pose_ok
                & (f_touch > self._eff_touch_pad)
                & (lift < self.touch_lift_max)
                & on_table
            )
            info_out["success"] = ok
            info_out["touch_f_finger"] = f_touch
        else:
            raise RuntimeError(f"Unhandled verb {self.verb!r}")

        if self._ood_dual_eval is not None:
            dual = self.compute_ood_dual_success(**self._ood_dual_eval)
            info_out["success_first_axis"] = dual["success_first_axis"]
            info_out["success_second_axis"] = dual["success_second_axis"]

        return info_out

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info)
