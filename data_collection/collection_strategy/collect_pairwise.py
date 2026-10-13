#!/usr/bin/env python3
"""
Collect VerbObjectColor-v1 demos for pairwise comparison (factor grids 6 / 12 / 18).

Strategies:
  random   — sample K primary index pairs (K=3/6/9 for factor 6/12/18) in the grid, then i.i.d. uniform per episode.
  stair    — fixed staircase pattern (see pairwise_factor_patterns.py). Paper strategy "V" (ours).
  Lrandom — fixed lists for 6/12; 18 = row0 ∪ col0 + 3 random interior cells once per collect (seeded).

Default ``--num-demos`` is 300 (total successful MP trajectories). Third factor pool and distractor
count default to 2 (``--third-pool-size 2``, ``--num-distractors 2``).
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collection_strategy.lib.pairwise_factor_patterns import (
    primary_pair_from_indices,
    random_support_indices,
    structured_support,
)
from collection_strategy.lib.pairwise_strategies import Experiment, Strategy
from collection_strategy.lib.training_vocab import (
    THIRD_COLORS_FOR_VERB_OBJECT,
    THIRD_OBJECTS_FOR_VERB_COLOR,
    THIRD_VERBS_FOR_COLOR_OBJECT,
)
from scripts.run_verb_color_shape_motion_planning import run_verb_color_shape_motion_planning


def _third_factor_pool(experiment: Experiment, pool_size: int) -> tuple[str, ...]:
    """First ``pool_size`` entries of the third-factor list for this experiment (min 1, max full length)."""
    if experiment == "verb_color":
        full = THIRD_OBJECTS_FOR_VERB_COLOR
    elif experiment == "verb_object":
        full = THIRD_COLORS_FOR_VERB_OBJECT
    elif experiment == "color_object":
        full = THIRD_VERBS_FOR_COLOR_OBJECT
    else:
        raise ValueError(experiment)
    n = max(1, min(int(pool_size), len(full)))
    return full[:n]


def _materialize_full(
    experiment: Experiment,
    pair: tuple,
    rng: random.Random,
    third_pool_size: int,
) -> tuple[str, str, str]:
    """Return (verb, color, shape) for env; third factor sampled uniformly from the first ``third_pool_size`` pool entries."""
    pool = _third_factor_pool(experiment, third_pool_size)
    if experiment == "verb_color":
        v, c = pair
        s = rng.choice(pool)
        return v, c, s
    if experiment == "verb_object":
        v, sh = pair
        c = rng.choice(pool)
        return v, c, sh
    if experiment == "color_object":
        c, sh = pair
        v = rng.choice(pool)
        return v, c, sh
    raise ValueError(experiment)


def _infinite_shuffled_primary_pairs(
    base_pairs: list[tuple],
    rng: random.Random,
):
    """Yield primary pairs forever: repeated full passes, each pass a fresh shuffle."""
    b = base_pairs
    if not b:
        raise ValueError("empty pair support")
    while True:
        cyc = list(b)
        rng.shuffle(cyc)
        yield from cyc


def _existing_demo_indices(record_dir: Path, strategy: str) -> list[int]:
    """Return sorted existing raw demo indices for this strategy under record_dir."""
    pat = re.compile(rf"^{re.escape(strategy)}_(\d+)_")
    idxs: set[int] = set()
    for pth in record_dir.rglob("*.h5"):
        stem = pth.stem
        if stem.endswith("_rgb"):
            continue
        m = pat.match(stem)
        if m is None:
            continue
        idxs.add(int(m.group(1)))
    return sorted(idxs)


def _build_primary_support(
    experiment: Experiment,
    strategy: Strategy,
    factor_count: int,
    rng: random.Random,
) -> list[tuple]:
    if strategy == "random":
        idxs = random_support_indices(factor_count, rng)
        return [primary_pair_from_indices(experiment, ij) for ij in idxs]
    if strategy in ("stair", "Lrandom"):
        need_rng = strategy == "Lrandom" and factor_count == 18
        return structured_support(
            experiment,
            strategy,
            factor_count,
            rng if need_rng else None,
        )
    raise ValueError(strategy)


def main() -> None:
    p = argparse.ArgumentParser(description="Pairwise VerbObjectColor demo collection")
    p.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=("verb_color", "verb_object", "color_object"),
    )
    p.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=("random", "stair", "Lrandom"),
    )
    p.add_argument(
        "--factor-count",
        type=int,
        default=6,
        choices=(6, 12, 18),
        help="Primary factor grid: 6→2×2 (or 3×3 for stair), 12→4×4, 18→6×6 (see README).",
    )
    p.add_argument(
        "--record-base",
        type=str,
        default="outputs/demo_gen",
        help="Root directory; demos go to <record-base>/<experiment>/<record-segment>/",
    )
    p.add_argument(
        "--record-segment",
        type=str,
        default=None,
        metavar="NAME",
        help="Subfolder under <record-base>/<experiment>/ (default: same as --strategy). "
        "Use e.g. random_f6 vs random_f12 so different factor grids do not overwrite.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fixed-index",
        type=int,
        default=0,
        help="Reserved for future use (unused in current factor-grid strategies).",
    )
    p.add_argument(
        "--num-demos",
        type=int,
        default=300,
        help="Target successful MP demos on disk (default 300).",
    )
    p.add_argument(
        "--num-traj",
        type=int,
        default=1,
        help="Per scheduled episode: number of successful MP trajectories to record (when --only-count-success, "
        "the inner loop retries until this many successes; default 1).",
    )
    p.add_argument(
        "--only-count-success",
        dest="only_count_success",
        action="store_true",
        help="Retry motion planning until num_traj successes per episode (failed solves are not counted).",
    )
    p.add_argument(
        "--no-only-count-success",
        dest="only_count_success",
        action="store_false",
        help="Count failed MP attempts as completed episodes (not recommended).",
    )
    p.set_defaults(only_count_success=True)
    p.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        metavar="N",
        help="With only-count-success: max total MP attempts per scheduled episode; on exhaustion skip that episode (0 = unlimited).",
    )
    p.add_argument(
        "--sim-backend",
        type=str,
        default="cpu",
        help="Simulation backend (default cpu; use gpu for speed if supported).",
    )
    p.add_argument(
        "--render-backend",
        type=str,
        default="cpu",
        help="Render backend (default cpu; use gpu for speed if supported).",
    )
    p.add_argument(
        "--save-video",
        dest="save_video",
        action="store_true",
        help="Save motion-planning .mp4 videos under motionplanning/ (default: off).",
    )
    p.add_argument("--no-save-video", dest="save_video", action="store_false")
    p.set_defaults(save_video=False)
    p.add_argument("--vis", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--obj-xy-range",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=None,
        help="Sample object xy each episode from [xmin,xmax]×[ymin,ymax] (passed to motion planning).",
    )
    p.add_argument(
        "--third-pool-size",
        type=int,
        default=2,
        metavar="N",
        help="Third factor: sample uniformly from the first N values in README order (default 2).",
    )
    p.add_argument(
        "--num-distractors",
        type=int,
        default=2,
        metavar="N",
        help="Exact number of distractors per episode (0–3; default 2).",
    )
    p.add_argument(
        "--distractor-max",
        type=int,
        default=2,
        metavar="N",
        help="Caps distractor sampling in env (default 2; must be >= num-distractors).",
    )
    args = p.parse_args()

    experiment: Experiment = args.experiment  # type: ignore[assignment]
    strategy: Strategy = args.strategy  # type: ignore[assignment]
    factor_count = int(args.factor_count)

    rng = random.Random(args.seed)

    path_seg = args.record_segment or strategy
    record_dir = Path(args.record_base) / experiment / path_seg
    record_dir.mkdir(parents=True, exist_ok=True)

    distractor_max = max(0, min(3, int(args.distractor_max)))
    num_distractors = max(0, min(3, int(args.num_distractors)))
    if num_distractors > distractor_max:
        p.error("--num-distractors must be <= --distractor-max")

    obj_xy_range = None
    if args.obj_xy_range is not None:
        x0, x1, y0, y1 = args.obj_xy_range
        obj_xy_range = ((float(x0), float(x1)), (float(y0), float(y1)))

    n = max(1, args.num_demos)
    existing_indices = _existing_demo_indices(record_dir, strategy)
    success_count = len(existing_indices)
    next_demo_idx = (existing_indices[-1] + 1) if existing_indices else 0
    attempt_id = next_demo_idx
    if success_count > 0:
        print(
            f"Resume detected: found {success_count} existing raw demos under {record_dir}; "
            f"next idx={next_demo_idx}, target={n}."
        )
    if success_count >= n:
        print(f"Already have {success_count} demos (target {n}); nothing to collect.")
        return

    support = _build_primary_support(experiment, strategy, factor_count, rng)
    if not support:
        raise RuntimeError("empty support")

    if strategy == "random":

        def pick_pair() -> tuple:
            return rng.choice(support)

    else:
        pair_iter = _infinite_shuffled_primary_pairs(support, rng)

        def pick_pair() -> tuple:
            return next(pair_iter)

    while success_count < n:
        pair = pick_pair()
        verb, color, shape = _materialize_full(
            experiment, pair, rng, args.third_pool_size
        )
        traj_name = f"{strategy}_{next_demo_idx:04d}_{verb}_{color}_{shape}"
        out = run_verb_color_shape_motion_planning(
            verb,
            color,
            shape,
            num_traj=args.num_traj,
            only_count_success=args.only_count_success,
            max_attempts=(None if args.max_attempts <= 0 else args.max_attempts),
            seed_start=args.seed + attempt_id,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            save_video=args.save_video,
            vis=args.vis,
            record_dir=str(record_dir),
            debug=args.debug,
            distractor_max=distractor_max,
            num_distractors=num_distractors,
            traj_name=traj_name,
            obj_xy_range=obj_xy_range,
        )
        attempt_id += 1
        if out is None:
            print(
                f"WARNING: MP failed for attempt {attempt_id} ({traj_name}); trying new task.",
                file=sys.stderr,
            )
        else:
            success_count += 1
            next_demo_idx += 1


if __name__ == "__main__":
    main()
