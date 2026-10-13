"""
Replay ManiSkill trajectories (entry point for python -m mani_skill.trajectory.replay_trajectory).
Implementation lives in export_maniskill_trajectory to avoid duplicate code.
"""

from mani_skill.trajectory.export_maniskill_trajectory import main, parse_args

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn")
    main(parse_args())
