#!/usr/bin/env python3
"""
Collect VerbObjectColor-v1 demos for **conflict** sweeps (repo: ``conflict_experiment/``, sibling of ``collection_strategy/``).

Primary factors are **fixed** to the full index diagonal **(k,k)** for ``k = 0 .. n-1`` (``n = min`` of the two
primary axis lengths in ``training_vocab``, currently **6** for all three experiments; no factor-count flag):

  verb_color:   (color_idx, verb_idx) ∈ {(0,0), (1,1), …, (5,5)}
  verb_object:  (shape_idx, verb_idx) ∈ {(0,0), …, (5,5)}
  color_object: (color_idx, shape_idx) ∈ {(0,0), …, (5,5)}

Third factor: uniform over values at list indices given by ``--third-indices`` (default ``0,1``;
append more integers, e.g. ``0,1,2,4``).

Trajectory stem prefix: ``conflict`` (e.g. ``conflict_0003_lift_red_cube_rgb``).

Demo count: ``--num-demos`` (pipeline may pass oversampled count from shell).
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

# data_collection/ (repo root)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conflict_experiment.lib.conflict_sampling import (
    fixed_diagonal_primary_pairs,
    parse_third_indices,
    third_pool_for_indices,
)
from collection_strategy.lib.pairwise_factor_patterns import Experiment
from scripts.run_verb_color_shape_motion_planning import run_verb_color_shape_motion_planning

_TRAJ_PREFIX = "conflict"


def _materialize(
    experiment: Experiment,
    pair: tuple[str, ...],
    rng: random.Random,
    third_pool: tuple[str, ...],
) -> tuple[str, str, str]:
    if not third_pool:
        raise ValueError("empty third pool")
    if experiment == "verb_color":
        v, c = pair
        s = rng.choice(third_pool)
        return v, c, s
    if experiment == "verb_object":
        v, sh = pair
        c = rng.choice(third_pool)
        return v, c, sh
    if experiment == "color_object":
        c, sh = pair
        v = rng.choice(third_pool)
        return v, c, sh
    raise ValueError(experiment)


def _infinite_shuffled_pairs(
    base_pairs: list[tuple[str, ...]],
    rng: random.Random,
):
    b = base_pairs
    if not b:
        raise ValueError("empty primary support")
    while True:
        cyc = list(b)
        rng.shuffle(cyc)
        yield from cyc


def _existing_demo_indices(record_dir: Path) -> list[int]:
    pat = re.compile(rf"^{re.escape(_TRAJ_PREFIX)}_(\d+)_")
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


def main() -> None:
    p = argparse.ArgumentParser(description="Conflict experiment: full index diagonal (k,k) + third indices")
    p.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=("verb_color", "verb_object", "color_object"),
    )
    p.add_argument(
        "--record-base",
        type=str,
        default="outputs/demo_gen",
        help="Root; demos under <record-base>/<experiment>/<record-segment>/",
    )
    p.add_argument(
        "--record-segment",
        type=str,
        default=None,
        metavar="NAME",
        help="Subfolder under <record-base>/<experiment>/ (default: conflict).",
    )
    p.add_argument(
        "--third-indices",
        type=str,
        default="0,1",
        help="Comma-separated indices into THIRD_* list for this experiment (default 0,1).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-demos", type=int, default=300)
    p.add_argument("--num-traj", type=int, default=1)
    p.add_argument("--only-count-success", dest="only_count_success", action="store_true")
    p.add_argument("--no-only-count-success", dest="only_count_success", action="store_false")
    p.set_defaults(only_count_success=True)
    p.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        metavar="N",
        help="With only-count-success: max MP attempts per episode (0 = unlimited).",
    )
    p.add_argument("--sim-backend", type=str, default="cpu")
    p.add_argument("--render-backend", type=str, default="cpu")
    p.add_argument("--save-video", dest="save_video", action="store_true")
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
    )
    p.add_argument("--num-distractors", type=int, default=2, metavar="N")
    p.add_argument("--distractor-max", type=int, default=2, metavar="N")
    args = p.parse_args()

    experiment: Experiment = args.experiment  # type: ignore[assignment]
    rng = random.Random(args.seed)

    third_idx_list = parse_third_indices(args.third_indices)
    third_pool = third_pool_for_indices(experiment, third_idx_list)

    path_seg = args.record_segment or _TRAJ_PREFIX
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
    existing_indices = _existing_demo_indices(record_dir)
    success_count = len(existing_indices)
    next_demo_idx = (existing_indices[-1] + 1) if existing_indices else 0
    attempt_id = next_demo_idx
    if success_count > 0:
        print(
            f"Resume: {success_count} existing raw demos under {record_dir}; "
            f"next idx={next_demo_idx}, target={n}."
        )
    if success_count >= n:
        print(f"Already have {success_count} demos (target {n}); nothing to collect.")
        return

    support = fixed_diagonal_primary_pairs(experiment)
    pair_iter = _infinite_shuffled_pairs(support, rng)

    while success_count < n:
        pair = next(pair_iter)
        verb, color, shape = _materialize(experiment, pair, rng, third_pool)
        traj_name = f"{_TRAJ_PREFIX}_{next_demo_idx:04d}_{verb}_{color}_{shape}"
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
