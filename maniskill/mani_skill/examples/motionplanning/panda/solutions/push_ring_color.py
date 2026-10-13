"""Motion planning for PushRingColor-v1: push ring so x increases (forward), y roughly same."""
import numpy as np
import sapien

from mani_skill.envs.tasks import PushRingColorEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: PushRingColorEnv, seed=None, debug=False, vis=False, options=None):
    env.reset(seed=seed, options=options or {})
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped
    planner.close_gripper()

    init_xy = env._obj_init_xy[0].cpu().numpy()
    obj_z = float(env.obj.pose.p[0, 2].cpu().numpy())
    tcp_q = env.agent.tcp.pose.sp.q

    reach_pose = sapien.Pose(
        p=env.obj.pose.sp.p + np.array([-0.05, 0, 0.1]),
        q=tcp_q,
    )
    planner.move_to_pose_with_screw(reach_pose)

    reach_pose = sapien.Pose(
        p=env.obj.pose.sp.p + np.array([-0.05, 0, 0]),
        q=tcp_q,
    )
    planner.move_to_pose_with_screw(reach_pose)

    push_distance = 0.12
    goal_pose = sapien.Pose(
        p=np.array([init_xy[0] + push_distance, init_xy[1], obj_z]),
        q=tcp_q,
    )
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
