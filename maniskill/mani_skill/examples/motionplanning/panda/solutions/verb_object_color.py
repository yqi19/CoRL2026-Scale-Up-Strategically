"""Motion planning for VerbObjectColor-v1 (single env, batched dim 0)."""
import numpy as np
import sapien
import transforms3d.quaternions as tq
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.verb_object_color_env import VerbObjectColorEnv
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def _np_p(p):
    return p[0].cpu().numpy() if hasattr(p, "cpu") else np.array(p)


def _np_q(q):
    return q[0].cpu().numpy() if hasattr(q, "cpu") else np.array(q)


def _move_pose_screw_or_rrt(planner: PandaArmMotionPlanningSolver, pose: sapien.Pose):
    """Screw first (fast); RRTConnect if screw fails (helps narrow / mesh-heavy grasps)."""
    r = planner.move_to_pose_with_screw(pose)
    if r == -1:
        r = planner.move_to_pose_with_RRTConnect(pose)
    return r


# Tall or tapering meshes: small rotate-lift clears the table poorly during wrist twist → slip / plan fail.
_ROTATE_EXTRA_LIFT_SHAPES = frozenset({"pyramid", "star"})


def solve(env, seed=None, debug=False, vis=False, options=None):
    env.reset(seed=seed, options=options or {})
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    u = env.unwrapped
    assert isinstance(u, VerbObjectColorEnv)
    verb = u.verb
    FINGER_LENGTH = 0.028

    if verb in ("lift", "rotate", "slide", "grasp", "touch"):
        obb = get_actor_obb(u.obj)
        approaching = np.array([0, 0, -1])
        target_closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        finger_depth = FINGER_LENGTH
        # Pyramid/star OBB grasps sit high on sloped sides; a bit more depth uses wider lower bulk.
        if u.object_shape in _ROTATE_EXTRA_LIFT_SHAPES and verb == "rotate":
            finger_depth = FINGER_LENGTH + 0.006
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=finger_depth,
        )
        closing, grasp_center = grasp_info["closing"], grasp_info["center"]
        # Use OBB-derived TCP point (not actor pose.p); important for cup/car/pyramid meshes.
        grasp_pose = u.agent.build_grasp_pose(approaching, closing, grasp_center)
    else:
        grasp_pose = None

    # --- push / pull / nudge: closed gripper from the side (no OBB grasp sequence).
    # slide is grasp-then-+y like lift/rotate; it opens/closes in its own branch — do not close here
    # or the demo starts with a spurious close→open.
    if verb in ("push", "pull", "nudge"):
        planner.close_gripper()
        init_xy = u._obj_init_xy[0].cpu().numpy()
        obj_z = float(u.obj.pose.p[0, 2].cpu().numpy())
        tcp_q = u.agent.tcp.pose.sp.q

    res = None

    if verb == "lift":
        planner.open_gripper()
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_screw(reach_pose)
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        # Straight vertical lift only; TCP rise comfortably above env pick height threshold.
        tcp_p = _np_p(u.agent.tcp.pose.p)
        lift_h = float(u._eff_pick_height_thresh) + 0.07
        up_pose = sapien.Pose(
            p=tcp_p + np.array([0.0, 0.0, lift_h]),
            q=grasp_pose.q,
        )
        res = planner.move_to_pose_with_screw(up_pose)

    elif verb == "push":
        reach_pose = sapien.Pose(
            p=_np_p(u.obj.pose.p) + np.array([-0.05, 0, 0.1]),
            q=tcp_q,
        )
        planner.move_to_pose_with_screw(reach_pose)
        reach_pose = sapien.Pose(
            p=_np_p(u.obj.pose.p) + np.array([-0.05, 0, 0]),
            q=tcp_q,
        )
        planner.move_to_pose_with_screw(reach_pose)
        goal_pose = sapien.Pose(
            p=np.array([init_xy[0] + 0.085, init_xy[1], obj_z]),
            q=tcp_q,
        )
        res = planner.move_to_pose_with_screw(goal_pose)

    elif verb == "nudge":
        reach_pose = sapien.Pose(
            p=_np_p(u.obj.pose.p) + np.array([-0.05, 0, 0.1]),
            q=tcp_q,
        )
        planner.move_to_pose_with_screw(reach_pose)
        reach_pose = sapien.Pose(
            p=_np_p(u.obj.pose.p) + np.array([-0.05, 0, 0]),
            q=tcp_q,
        )
        planner.move_to_pose_with_screw(reach_pose)
        goal_pose = sapien.Pose(
            p=np.array([init_xy[0] + 0.048, init_xy[1], obj_z]),
            q=tcp_q,
        )
        res = planner.move_to_pose_with_screw(goal_pose)

    elif verb == "pull":
        reach_pose = sapien.Pose(
            p=_np_p(u.obj.pose.p) + np.array([0.05, 0, 0.1]),
            q=tcp_q,
        )
        planner.move_to_pose_with_screw(reach_pose)
        reach_pose = sapien.Pose(
            p=_np_p(u.obj.pose.p) + np.array([0.05, 0, 0]),
            q=tcp_q,
        )
        planner.move_to_pose_with_screw(reach_pose)
        goal_pose = sapien.Pose(
            p=np.array([init_xy[0] - 0.085, init_xy[1], obj_z]),
            q=tcp_q,
        )
        res = planner.move_to_pose_with_screw(goal_pose)

    elif verb == "slide":
        # Grasp, then move +y (see env success: +y displacement)
        planner.open_gripper()
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_screw(reach_pose)
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        gp = _np_p(grasp_pose.p)
        goal_pose = sapien.Pose(
            p=np.array([gp[0], gp[1] + 0.14, gp[2]]),
            q=grasp_pose.q,
        )
        res = planner.move_to_pose_with_screw(goal_pose)

    elif verb == "rotate":
        # Small vertical lift, then 90° world-z at fixed TCP. Taller/tapered shapes need more clearance.
        planner.open_gripper()
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_screw(reach_pose)
        planner.move_to_pose_with_screw(grasp_pose)
        # Tapering meshes: give fingers longer to squeeze before lift+twist.
        planner.close_gripper(t=10 if u.object_shape in _ROTATE_EXTRA_LIFT_SHAPES else 6)
        gp = _np_p(grasp_pose.p)
        gq = grasp_pose.q
        lift_h = 0.075 if u.object_shape in _ROTATE_EXTRA_LIFT_SHAPES else 0.045
        lift_pose = sapien.Pose(p=gp + np.array([0.0, 0.0, lift_h]), q=gq)
        res = _move_pose_screw_or_rrt(planner, lift_pose)
        if res != -1:
            # Two 45° world-z steps (pyramid/star): smaller single rotation often plans more reliably than one 90°.
            if u.object_shape in _ROTATE_EXTRA_LIFT_SHAPES:
                qz45 = euler2quat(0, 0, np.pi / 4)
                for _ in range(2):
                    tcp_p = _np_p(u.agent.tcp.pose.p)
                    tcp_q = _np_q(u.agent.tcp.pose.q)
                    q_goal = tq.qmult(qz45, tcp_q)
                    rot_pose = sapien.Pose(p=tcp_p, q=q_goal)
                    res = _move_pose_screw_or_rrt(planner, rot_pose)
                    if res == -1:
                        break
            else:
                tcp_p = _np_p(u.agent.tcp.pose.p)
                tcp_q = _np_q(u.agent.tcp.pose.q)
                qz90 = euler2quat(0, 0, np.pi / 2)
                q_goal = tq.qmult(qz90, tcp_q)
                rot_pose = sapien.Pose(p=tcp_p, q=q_goal)
                res = _move_pose_screw_or_rrt(planner, rot_pose)

    elif verb == "grasp":
        planner.open_gripper()
        op = _np_p(u.obj.pose.p)
        tcp_q_down = u.agent.tcp.pose.sp.q
        pre = sapien.Pose(p=op + np.array([0, 0, 0.14]), q=tcp_q_down)
        planner.move_to_pose_with_screw(pre)
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        oz = float(u.obj.pose.p[0, 2].cpu())
        settle_grasp = sapien.Pose(
            p=np.array([op[0], op[1], oz + 0.004]),
            q=grasp_pose.q,
        )
        res = planner.move_to_pose_with_screw(settle_grasp)

    elif verb == "touch":
        planner.open_gripper()
        op = _np_p(u.obj.pose.p)
        tcp_q_down = u.agent.tcp.pose.sp.q
        pre = sapien.Pose(p=op + np.array([0, 0, 0.13]), q=tcp_q_down)
        planner.move_to_pose_with_screw(pre)
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        oz = float(u.obj.pose.p[0, 2].cpu())
        settle = sapien.Pose(
            p=np.array([op[0], op[1], oz + 0.022]),
            q=grasp_pose.q,
        )
        res = planner.move_to_pose_with_screw(settle)

    planner.close()
    return res
