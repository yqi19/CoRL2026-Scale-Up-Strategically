#!/usr/bin/env python3
"""Collect color_spatial demos: color × spatial position (fixed verb=lift, shape=cube)."""
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
    LRANDOM_6,
    LRANDOM_12,
    Lrandom_18_index_support,
    STAIR_6,
    STAIR_12,
    STAIR_18,
    random_support_indices,
)
from collection_strategy.lib.training_vocab import TRAINING_COLORS
from scripts.run_verb_color_shape_motion_planning import run_verb_color_shape_motion_planning

SPATIAL_SET: tuple[str, ...] = ("left", "right", "middle", "front", "behind", "middle")


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


def _infinite_shuffled_pairs(base_pairs: list[tuple[str, str]], rng: random.Random):
    if not base_pairs:
        raise ValueError("empty pair support")
    while True:
        cyc = list(base_pairs)
        rng.shuffle(cyc)
        yield from cyc


def _primary_pair(ij: tuple[int, int]) -> tuple[str, str]:
    i, j = ij
    return TRAINING_COLORS[i], SPATIAL_SET[j]


def _spatial_xy(spatial: str, rng: random.Random) -> list[float]:
    anchors = {
        "left":   (-0.10, 0.00),
        "right":  ( 0.10, 0.00),
        "middle": ( 0.00, 0.00),
        "front":  ( 0.00, 0.10),
        "behind": ( 0.00,-0.10),
    }
    ax, ay = anchors[spatial]
    return [ax + rng.uniform(-0.012, 0.012), ay + rng.uniform(-0.012, 0.012)]


def _build_support(strategy: str, factor_count: int, rng: random.Random) -> list[tuple[str, str]]:
    if strategy == "random":
        idxs = random_support_indices(factor_count, rng)
    elif strategy == "Lrandom":
        if factor_count == 6:
            idxs = list(LRANDOM_6)
        elif factor_count == 12:
            idxs = list(LRANDOM_12)
        else:
            idxs = Lrandom_18_index_support(rng)
    elif strategy == "stair":
        idxs = STAIR_6 if factor_count == 6 else STAIR_12 if factor_count == 12 else STAIR_18
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")
    return [_primary_pair(ij) for ij in idxs]


def main() -> None:
    p = argparse.ArgumentParser(description="color_spatial demo collection (color × spatial position)")
    p.add_argument("--strategy", required=True, choices=("random", "Lrandom", "stair"))
    p.add_argument("--factor-count", type=int, default=12, choices=(6, 12, 18))
    p.add_argument("--record-base", type=str, default="outputs/demo_gen")
    p.add_argument("--record-segment", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-demos", type=int, default=200)
    p.add_argument("--num-traj", type=int, default=1)
    p.add_argument("--only-count-success", dest="only_count_success", action="store_true")
    p.add_argument("--no-only-count-success", dest="only_count_success", action="store_false")
    p.set_defaults(only_count_success=True)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--sim-backend", type=str, default="cpu")
    p.add_argument("--render-backend", type=str, default="cpu")
    p.add_argument("--save-video", dest="save_video", action="store_true")
    p.add_argument("--no-save-video", dest="save_video", action="store_false")
    p.set_defaults(save_video=False)
    p.add_argument("--vis", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--num-distractors", type=int, default=2)
    p.add_argument("--distractor-max", type=int, default=3)
    args = p.parse_args()

    strategy = args.strategy
    factor_count = int(args.factor_count)
    rng = random.Random(args.seed)

    segment = args.record_segment or f"{strategy}_f{factor_count}"
    record_dir = Path(args.record_base) / "color_spatial" / segment
    record_dir.mkdir(parents=True, exist_ok=True)

    support = _build_support(strategy, factor_count, rng)

    if strategy == "random":
        def pick_pair() -> tuple[str, str]:
            return rng.choice(support)
    else:
        pair_iter = _infinite_shuffled_pairs(support, rng)
        def pick_pair() -> tuple[str, str]:
            return next(pair_iter)

    n = max(1, int(args.num_demos))
    existing = _existing_demo_indices(record_dir, strategy)
    success_count = len(existing)
    next_idx = (existing[-1] + 1) if existing else 0
    attempt_id = next_idx

    if success_count > 0:
        print(f"Resume: found {success_count} existing demos; next_idx={next_idx}, target={n}.")
    if success_count >= n:
        print(f"Already have {success_count} demos (target {n}); nothing to collect.")
        return

    while success_count < n:
        color, spatial = pick_pair()
        attr = f"{color}-{spatial}"
        traj_name = f"{strategy}_{next_idx:04d}_lift_{attr}_cube"

        out = run_verb_color_shape_motion_planning(
            verb="lift",
            color=color,
            shape="cube",
            obj_xy=_spatial_xy(spatial, rng),
            object_size_jiggle=0.0,
            traj_name=traj_name,
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
            distractor_max=max(0, min(3, int(args.distractor_max))),
            num_distractors=max(0, min(3, int(args.num_distractors))),
        )
        attempt_id += 1
        if out is not None:
            success_count += 1
            next_idx += 1


if __name__ == "__main__":
    main()
