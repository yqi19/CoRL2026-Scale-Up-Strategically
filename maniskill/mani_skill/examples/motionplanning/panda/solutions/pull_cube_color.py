"""Motion planning for PullCubeColor-v1: pull cube so x decreases (backward), y roughly same. No goal_region."""
import numpy as np
import sapien

from mani_skill.envs.tasks import PullCubeColorEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: PullCubeColorEnv, seed=None, debug=False, vis=False, options=None):
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

    # Initial cube xy (task: pull so x decreases, y stays ~same)
    init_xy = env._obj_init_xy[0].cpu().numpy()
    obj_z = float(env.obj.pose.p[0, 2].cpu().numpy())
    tcp_q = env.agent.tcp.pose.sp.q

    # Approach to the z positive side of the cube
    reach_pose = sapien.Pose(
        p=env.obj.pose.sp.p + np.array([0.05, 0, 0.1]),
        q=tcp_q,
    )
    planner.move_to_pose_with_screw(reach_pose)

    # Approach from +x side of the cube (pull toward -x)
    reach_pose = sapien.Pose(
        p=env.obj.pose.sp.p + np.array([0.05, 0, 0]),
        q=tcp_q,
    )
    planner.move_to_pose_with_screw(reach_pose)

    # Pull toward -x: move TCP to (init_x - pull_distance, init_y, z)
    pull_distance = 0.12
    goal_pose = sapien.Pose(
        p=np.array([init_xy[0] - pull_distance, init_xy[1], obj_z]),
        q=tcp_q,
    )
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
