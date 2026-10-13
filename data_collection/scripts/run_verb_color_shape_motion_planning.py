#!/usr/bin/env python3
"""
Motion planning for Verb × Color × Shape tabletop envs.

Legacy (unchanged env ids):  cube/sphere + red/yellow/blue + pick/push/pull → *CubeColor-v1 / *SphereColor-v1.

Extended:  VerbObjectColor-v1 — verbs lift|pick|pick_up, grasp|press, push, pull, rotate, slide, nudge, touch;
           colors red,yellow,blue,orange,green,black; shapes cube,sphere,cup,ring,car,pyramid,star (cup/car from ``asset/*.obj``);
           ``--task-difficulty`` > 1 tightens success / layout. Aliases: can|cylinder|mug→cup.

Output: demos/<env_id>/motionplanning/

CLI:
  python scripts/run_verb_color_shape_motion_planning.py --verb push --color orange --shape ring
  python scripts/run_verb_color_shape_motion_planning.py --verb rotate --color black --shape cube --distractor-max 0
  python scripts/visualize_verb_object_samples.py --verb touch --shape cup -n 2

If mplib segfaults: -b cpu --render-backend cpu.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
maniskill_root = os.path.dirname(script_dir)
if maniskill_root not in sys.path:
    sys.path.insert(0, maniskill_root)

import mani_skill.envs  # noqa: F401
from mani_skill.examples.motionplanning.panda.run import MP_SOLUTIONS
from mani_skill.utils.wrappers.record import RecordEpisode

# Legacy: ring + 3 colors + pick/push/pull only (original *Color-v1 envs)
LEGACY_VERBS = ("pick", "push", "pull")
LEGACY_COLORS = ("red", "yellow", "blue")
# Legacy tabletop *Color-v1 envs (ring always uses VerbObjectColor-v1 below).
LEGACY_SHAPES = ("cube", "sphere")

# Extended: VerbObjectColor-v1 (8 verbs × 6 colors × 7 shapes; distractors 0–3)
EXT_VERBS = ("lift", "grasp", "push", "pull", "rotate", "slide", "nudge", "touch")
EXT_COLORS = ("red", "yellow", "blue", "orange", "green", "black")
EXT_SHAPES = ("cube", "sphere", "cup", "ring", "car", "pyramid", "star")

COLOR_TO_ID = {name: i for i, name in enumerate(EXT_COLORS)}
COLOR_NAMES = {i: name for i, name in enumerate(EXT_COLORS)}

VERB_SHAPE_TO_ENV_ID = {
    ("pick", "cube"): "PickCubeColor-v1",
    ("pick", "sphere"): "PickSphereColor-v1",
    ("push", "cube"): "PushCubeColor-v1",
    ("push", "sphere"): "PushSphereColor-v1",
    ("pull", "cube"): "PullCubeColor-v1",
    ("pull", "sphere"): "PullSphereColor-v1",
}


def _legacy_verb_key(verb: str) -> str:
    """Normalize only for routing to legacy *Color-v1 envs (pick stays pick)."""
    v = verb.lower().strip().replace(" ", "_").replace("-", "_")
    if v in ("pick_up", "pickup"):
        return "pick"
    return v


def _verb_for_verb_object_color(verb: str) -> str:
    """CLI aliases for VerbObjectColor-v1 (pick→lift, press→grasp)."""
    v = _legacy_verb_key(verb)
    if v == "pick":
        return "lift"
    if v == "press":
        return "grasp"
    return v


def get_env_id_and_color(
    verb: str,
    color: str,
    shape: str,
    distractor_max: int = 3,
    task_difficulty: float = 1.0,
) -> tuple[str, int, dict]:
    """Returns (env_id, object_color_id, extra_make_kwargs). extra_make_kwargs empty for legacy."""
    v_key = _legacy_verb_key(verb)
    verb_voc = _verb_for_verb_object_color(verb)
    color = color.lower().strip()
    shape = shape.lower().strip()
    if shape in ("can", "cylinder", "mug"):
        shape = "cup"
    if color not in COLOR_TO_ID:
        raise ValueError(f"color must be one of {EXT_COLORS}, got {color!r}")
    color_id = COLOR_TO_ID[color]

    use_legacy = (
        shape in ("cube", "sphere")
        and color in LEGACY_COLORS
        and v_key in LEGACY_VERBS
    )
    if use_legacy:
        env_id = VERB_SHAPE_TO_ENV_ID[(v_key, shape)]
        return env_id, color_id, {}

    if verb_voc not in EXT_VERBS:
        raise ValueError(
            f"verb must be one of {EXT_VERBS} (or pick/pick_up/press), got {verb!r}"
        )
    if shape not in EXT_SHAPES:
        raise ValueError(
            f"shape must be one of {EXT_SHAPES} (legacy envs: cube/sphere + red/yellow/blue + pick/push/pull), got {shape!r}"
        )

    extra = dict(
        verb=verb_voc,
        object_shape=shape,
        object_color_id=color_id,
        distractor_max=max(0, min(3, int(distractor_max))),
        task_difficulty=float(task_difficulty),
    )
    return "VerbObjectColor-v1", color_id, extra


def run_verb_color_shape_motion_planning(
    verb: str,
    color: str,
    shape: str,
    *,
    obj_xy=None,
    obj_xy_range=None,
    num_traj: int = 1,
    only_count_success: bool = False,
    max_attempts: int | None = 3,
    seed_start: int = 0,
    sim_backend: str = "cpu",
    render_backend: str = "cpu",
    save_video: bool = True,
    vis: bool = False,
    record_dir: str = "demos",
    obs_mode: str = "none",
    object_sizes_path: str = None,
    traj_name: str = None,
    debug: bool = False,
    distractor_max: int = 3,
    num_distractors: int | None = None,
    object_size_jiggle: float = 0.2,
    task_difficulty: float = 1.0,
    target_size_scale: float | None = None,
    distractor_size_scales: list[float] | tuple[float, ...] | None = None,
    distractor_match_target_geometry: tuple[bool, ...] | None = None,
    **kwargs,
):
    """
    Run motion planning on a Verb×Color×Shape env (cube, sphere, ring).

    Args:
        verb: lift | grasp | push | pull | rotate | slide | nudge | touch (CLI: pick→lift, press→grasp)
        color: red | yellow | blue | orange | green | black
        shape: cube | sphere | cup | ring | car | pyramid | star
        obj_xy: optional [x, y] to fix object spawn
        obj_xy_range: optional ((x_min, x_max), (y_min, y_max)) to sample object xy
        num_traj: number of trajectories (or number of successful when only_count_success)
        only_count_success: if True, keep generating until num_traj successful demos
        max_attempts: if only_count_success, max total MP attempts per call (default 3). None or <=0 means unlimited.
        seed_start: starting seed
        sim_backend: cpu | gpu
        render_backend: cpu | gpu
        save_video: whether to save videos
        vis: show GUI
        record_dir: output base dir (writes to record_dir/<env_id>/motionplanning/)
        obs_mode: env obs_mode (default "none")
        object_sizes_path: optional path to .txt with object sizes (default: tabletop/object_sizes_default.txt)
        traj_name: base name for trajectory .h5/.json (default: timestamp like 20260315_123456). Videos will be 0.mp4, 1.mp4, ... in same dir.
        num_distractors: if set (0–3), passed to ``reset(..., options)`` so each episode uses exactly this many distractors.

    Returns:
        Path to the written ``.h5`` on success. If ``only_count_success`` and fewer than ``num_traj``
        successes after ``max_attempts``, returns ``None`` (partial files removed when possible).
    """
    env_id, color_id, extra_make = get_env_id_and_color(
        verb,
        color,
        shape,
        distractor_max=distractor_max,
        task_difficulty=task_difficulty,
    )
    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(f"No motion planning solver for {env_id}. Available: {list(MP_SOLUTIONS.keys())}")

    render_mode = "rgb_array"
    # Allow fully headless collection when render backend is disabled.
    if str(render_backend).lower() == "none":
        render_mode = None

    make_kw = dict(
        obs_mode=obs_mode,
        control_mode="pd_joint_pos",
        render_mode=render_mode,
        sim_backend=sim_backend,
        render_backend=render_backend,
    )
    if env_id == "VerbObjectColor-v1":
        make_kw.update(extra_make)
        make_kw["object_size_jiggle"] = float(object_size_jiggle)
        if target_size_scale is not None:
            make_kw["target_size_scale"] = float(target_size_scale)
        if distractor_size_scales is not None:
            make_kw["distractor_size_scales"] = [float(x) for x in distractor_size_scales]
        if distractor_match_target_geometry is not None:
            make_kw["distractor_match_target_geometry"] = tuple(
                bool(x) for x in distractor_match_target_geometry
            )
    else:
        make_kw["object_color_id"] = color_id
    if object_sizes_path is not None:
        make_kw["object_sizes_path"] = object_sizes_path
    env = gym.make(env_id, **make_kw)

    if traj_name is None:
        traj_name = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(record_dir, env_id, "motionplanning")
    os.makedirs(output_dir, exist_ok=True)
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name=traj_name,
        save_video=save_video,
        source_type="motionplanning",
        source_desc="run_verb_color_shape_motion_planning",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )
    output_h5_path = env._h5_file.filename

    if debug:
        env.unwrapped._debug_motionplan = True
    solve = MP_SOLUTIONS[env_id]
    rng = np.random.default_rng(seed_start)
    successes = []
    passed = 0
    attempt = 0
    target = num_traj
    max_attempts_cap: int | None = None if max_attempts is None or max_attempts <= 0 else int(max_attempts)
    pbar = tqdm(
        desc=f"{env_id} color={color}",
        total=target,
        unit="demo",
    )
    while True:
        if only_count_success:
            if passed >= num_traj:
                break
            if max_attempts_cap is not None and attempt >= max_attempts_cap:
                break
        else:
            if attempt >= num_traj:
                break
        options = {}
        if num_distractors is not None:
            options["num_distractors"] = int(num_distractors)
        if obj_xy_range is not None:
            (x_min, x_max), (y_min, y_max) = obj_xy_range
            x = float(rng.uniform(x_min, x_max))
            y = float(rng.uniform(y_min, y_max))
            options["obj_xy"] = [x, y]
        elif obj_xy is not None:
            options["obj_xy"] = [float(obj_xy[0]), float(obj_xy[1])]
        seed = seed_start + attempt
        try:
            res = solve(env, seed=seed, debug=debug, vis=vis, options=options)
        except Exception as e:
            print(f"Motion planning failed: {e}")
            res = -1

        if res == -1:
            success = False
        else:
            success = res[-1]["success"].item()

        successes.append(success)
        if success:
            env.flush_trajectory()
            if save_video:
                env.flush_video()
            passed += 1
            pbar.update(1)
        else:
            # Only flush if we have trajectory data (solve may have failed before any step)
            try:
                if getattr(env, "_trajectory_buffer", None) is not None:
                    env.flush_trajectory(save=False)
                    if save_video:
                        env.flush_video(save=False)
            except Exception:
                pass
            if not only_count_success:
                passed += 1
                pbar.update(1)
        pbar.set_postfix(
            success_rate=f"{sum(successes)}/{len(successes)}" if successes else "0/0",
            passed=passed,
        )
        attempt += 1

    pbar.close()
    env.close()
    if only_count_success and passed < num_traj:
        print(
            f"WARNING: Motion planning got {passed}/{num_traj} successes after {attempt} attempts "
            f"(max_attempts={max_attempts_cap}); skipping this episode.",
            file=sys.stderr,
        )
        try:
            hp = Path(output_h5_path)
            if hp.is_file():
                hp.unlink()
            jp = hp.with_suffix(".json")
            if jp.is_file():
                jp.unlink()
        except OSError as e:
            print(f"WARNING: could not remove partial trajectory files: {e}", file=sys.stderr)
        return None
    if successes:
        print(f"Success rate: {sum(successes)}/{len(successes)} ({100*sum(successes)/len(successes):.1f}%)")
    else:
        print("Success rate: 0/0")
    return output_h5_path


def main():
    parser = argparse.ArgumentParser(
        description="Motion planning: legacy ring+3colors, or extended VerbObjectColor-v1 (8 verbs × 6 colors × 7 shapes + distractors)."
    )
    parser.add_argument(
        "--verb",
        type=str,
        required=True,
        help="Verb: lift|pick|pick_up, grasp|press, push, pull, rotate, slide, nudge, touch",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="red",
        choices=EXT_COLORS,
        help="Object color",
    )
    parser.add_argument(
        "--shape",
        type=str,
        required=True,
        help="Shapes: cube, sphere, cup, ring, car, pyramid, star (aliases: can|cylinder|mug→cup)",
    )
    parser.add_argument(
        "--obj-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Fix object spawn at (x, y); omit for random sampling",
    )
    parser.add_argument(
        "--obj-xy-range",
        type=float,
        nargs=4,
        default=None,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help="Sample object xy in range (e.g. -0.1 0.1 -0.1 0.1)",
    )
    parser.add_argument(
        "-n", "--num-traj",
        type=int,
        default=1,
        help="Number of trajectories (successful when --only-count-success)",
    )
    parser.add_argument(
        "--only-count-success",
        action="store_true",
        help="Generate until num_traj successful demos (discard failures)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        metavar="N",
        help="With --only-count-success: max total MP attempts per run (default 3). Use 0 for unlimited.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Starting seed",
    )
    parser.add_argument(
        "-b", "--sim-backend",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Simulation backend",
    )
    parser.add_argument(
        "--render-backend",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Render backend (use cpu if mplib segfaults)",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Do not save videos (default: save videos)",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Show GUI",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="demos",
        help="Output directory (writes to <record-dir>/<env_id>/motionplanning/)",
    )
    parser.add_argument(
        "--object-sizes",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to .txt file with object sizes (default: use built-in object_sizes_default.txt)",
    )
    parser.add_argument(
        "--traj-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Base name for trajectory file (e.g. pick_cube_blue). Output: <name>.h5, <name>.json, 0.mp4, 1.mp4, ... (default: timestamp)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print progress before mplib.Planner() and other steps (to locate segfault; see ManiSkill #1100).",
    )
    parser.add_argument(
        "--distractor-max",
        type=int,
        default=3,
        help="VerbObjectColor-v1 only: max distractors (0–3). Episode samples actual count in [0, max].",
    )
    parser.add_argument(
        "--object-size-jiggle",
        type=float,
        default=0.2,
        metavar="J",
        help="VerbObjectColor-v1 only: each object scale ~ Uniform[1-J,1+J] at scene build (default 0.2). Use 0 to disable.",
    )
    parser.add_argument(
        "--task-difficulty",
        type=float,
        default=1.0,
        metavar="D",
        help="VerbObjectColor-v1: 1.0=baseline; >1 harder (larger required motion, tighter spawn, more jiggle scale).",
    )
    args = parser.parse_args()

    obj_xy_range = None
    if args.obj_xy_range is not None:
        x_min, x_max, y_min, y_max = args.obj_xy_range
        obj_xy_range = ((x_min, x_max), (y_min, y_max))

    out = run_verb_color_shape_motion_planning(
        verb=args.verb,
        color=args.color,
        shape=args.shape,
        obj_xy=args.obj_xy,
        obj_xy_range=obj_xy_range,
        num_traj=args.num_traj,
        only_count_success=args.only_count_success,
        max_attempts=args.max_attempts,
        seed_start=args.seed,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        save_video=not args.no_save_video,
        vis=args.vis,
        record_dir=args.record_dir,
        object_sizes_path=args.object_sizes,
        traj_name=args.traj_name,
        debug=args.debug,
        distractor_max=args.distractor_max,
        object_size_jiggle=args.object_size_jiggle,
        task_difficulty=args.task_difficulty,
    )
    if out is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
