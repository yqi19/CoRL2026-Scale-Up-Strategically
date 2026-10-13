#!/usr/bin/env python3
"""
Collect VerbObjectColor-v1 demos with **full** (verb, color, shape) triples — no third-factor pool.

Factor scaling (``--factor-count``):

- **15** → indices ``0..2`` per axis (3³ cube)
- **20** → ``0..3`` (4³)
- **30** → ``0..5`` (6³ full training vocab)

Strategies (see ``lib/triple_factor_patterns.py``):

- ``random`` — ``factor // 3`` index triples, greedy L∞ spread
- ``Lrandom`` — three axis rays ``(i,0,0)``, ``(0,j,0)``, ``(0,0,k)`` + extras (1 / 2 / 4 by side)
- ``stair`` — verb ray ``(i,0,0)`` + shape ray ``(0,0,k)`` + space diagonal ``(i,i,i)`` + 3D zigzag (verb→color→shape)

Trajectory stems: ``<strategy>_<dddd>_<verb>_<color>_<shape>`` (same as pairwise).
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

from collection_strategy.lib.triple_factor_patterns import build_support
from scripts.run_verb_color_shape_motion_planning import run_verb_color_shape_motion_planning

TripleStrategy = str  # random | Lrandom | stair


def _infinite_shuffled_triples(
    base: list[tuple[str, str, str]],
    rng: random.Random,
):
    if not base:
        raise ValueError("empty support")
    while True:
        cyc = list(base)
        rng.shuffle(cyc)
        yield from cyc


def _existing_demo_indices(record_dir: Path, strategy: str) -> list[int]:
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


def main() -> None:
    p = argparse.ArgumentParser(description="Verb × Color × Object triple collection (VerbObjectColor-v1)")
    p.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=("random", "Lrandom", "stair"),
    )
    p.add_argument(
        "--factor-count",
        type=int,
        required=True,
        choices=(15, 20, 30),
        help="15→3³ indices, 20→4³, 30→6³ full vocab per axis.",
    )
    p.add_argument("--record-base", type=str, default="outputs/demo_gen")
    p.add_argument(
        "--record-segment",
        type=str,
        default=None,
        metavar="NAME",
        help="Subfolder under verb_color_object/ (default: <strategy>_f<factor>).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-demos", type=int, default=300)
    p.add_argument("--num-traj", type=int, default=1)
    p.add_argument("--only-count-success", dest="only_count_success", action="store_true")
    p.add_argument("--no-only-count-success", dest="only_count_success", action="store_false")
    p.set_defaults(only_count_success=True)
    p.add_argument("--max-attempts", type=int, default=3, metavar="N")
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

    strategy: TripleStrategy = args.strategy  # type: ignore[assignment]
    factor_count = int(args.factor_count)
    rng = random.Random(args.seed)

    experiment = "verb_color_object"
    path_seg = args.record_segment or f"{strategy}_f{factor_count}"
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
            f"Resume: {success_count} existing raw demos under {record_dir}; "
            f"next idx={next_demo_idx}, target={n}."
        )
    if success_count >= n:
        print(f"Already have {success_count} demos (target {n}); nothing to collect.")
        return

    support = build_support(strategy, factor_count, rng)
    if not support:
        raise RuntimeError("empty support")

    if strategy == "random":

        def pick() -> tuple[str, str, str]:
            return rng.choice(support)

    else:
        it = _infinite_shuffled_triples(support, rng)

        def pick() -> tuple[str, str, str]:
            return next(it)

    while success_count < n:
        verb, color, shape = pick()
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
