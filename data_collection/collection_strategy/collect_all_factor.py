#!/usr/bin/env python3
"""
All-factor demos (verb × color × shape × spatial × size), isolated pipeline.

Output stems (7 underscore segments, no ``+``):
  ``{strategy}_{idx:04d}_{verb}_{color}_{shape}_{spatial}_{size}``

Strategies (see ``lib/all_factor_support_f50.py``): ``random`` supports **f12** / **f20** / **f40**;
``stairvss`` supports **f20** / **f40** / **f50**; ``Lrandom`` supports **f12** / **f50**.
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

from collection_strategy.lib.all_factor_scene import pack_verb_object_color_options
from collection_strategy.lib.all_factor_support_f50 import (
    ALL_FACTOR_SIZE_LABELS,
    build_support,
    parse_size_pool_labels,
    remap_support_size_pool,
)
from collection_strategy.lib.training_vocab import (
    TRAINING_COLORS,
    TRAINING_SHAPES,
    TRAINING_VERBS,
)
from scripts.run_verb_color_shape_motion_planning import run_verb_color_shape_motion_planning

ALL_FACTOR_STRATEGIES = ("random", "stairvss", "Lrandom")
FACTOR_COUNT_NON_RANDOM = 50
RANDOM_FACTOR_COUNTS = frozenset({12, 20, 40})
RANDOM_SUPPORT_LEN = {12: 12, 20: 5, 40: 10}
STAIR_FACTOR_COUNTS = frozenset({20, 40, 50})
LRANDOM_FACTOR_COUNTS = frozenset({12, 50})
ALL_FACTOR_SPATIALS: tuple[str, ...] = ("left", "right", "middle", "front", "behind")
ALL_FACTOR_SIZES: tuple[str, ...] = ALL_FACTOR_SIZE_LABELS


def _spatial_xy(spatial: str, rng: random.Random) -> list[float]:
    anchors = {
        "left": (-0.10, 0.00),
        "right": (0.10, 0.00),
        "middle": (0.00, 0.00),
        "center": (0.00, 0.00),
        "front": (0.00, 0.10),
        "behind": (0.00, -0.10),
    }
    if spatial not in anchors:
        raise ValueError(spatial)
    ax, ay = anchors[spatial]
    return [ax + rng.uniform(-0.012, 0.012), ay + rng.uniform(-0.012, 0.012)]


def _tuple_to_labels(t: tuple[int, int, int, int, int]) -> tuple[str, str, str, str, str]:
    v, c, sh, sp, sz = t
    return (
        TRAINING_VERBS[v],
        TRAINING_COLORS[c],
        TRAINING_SHAPES[sh],
        ALL_FACTOR_SPATIALS[sp],
        ALL_FACTOR_SIZES[sz],
    )


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


def _infinite_shuffled(rows: list[tuple[int, int, int, int, int]], rng: random.Random):
    if not rows:
        raise ValueError("empty support")
    while True:
        cyc = list(rows)
        rng.shuffle(cyc)
        yield from cyc


def main() -> None:
    p = argparse.ArgumentParser(description="All-factor (5D) tabletop demos — isolated from pairwise/conflict.")
    p.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=ALL_FACTOR_STRATEGIES,
    )
    p.add_argument("--factor-count", type=int, default=FACTOR_COUNT_NON_RANDOM)
    p.add_argument("--record-base", type=str, default="outputs/demo_gen")
    p.add_argument("--record-segment", type=str, default=None, metavar="NAME")
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
    p.add_argument("--num-distractors", type=int, default=2, metavar="N")
    p.add_argument("--distractor-max", type=int, default=3, metavar="N")
    p.add_argument(
        "--size-pool",
        type=str,
        default=",".join(ALL_FACTOR_SIZES),
        metavar="LABELS",
        help=(
            "Comma-separated subset of size labels; strategy support (f12 for random, f50 otherwise) is remapped so every tuple "
            "uses only these sizes (default: all six)."
        ),
    )
    p.add_argument(
        "--small-large-distractor-policy",
        type=str,
        default="bernoulli",
        choices=("bernoulli", "none", "fixed", "random_uniform"),
        help="For absolute small/large only: how distractor count is chosen (relative sizes unchanged).",
    )
    p.add_argument(
        "--small-large-no-distractor-prob",
        type=float,
        default=0.70,
        metavar="P",
        help="With policy=bernoulli: P(episode has no distractors) for small/large (default 0.70).",
    )
    p.add_argument(
        "--small-large-fixed-num-distractors",
        type=int,
        default=0,
        metavar="N",
        help="With policy=fixed: exact distractor count for small/large (clamped by cap and CLI max).",
    )
    args = p.parse_args()

    strategy = args.strategy
    fc = int(args.factor_count)
    if strategy == "random":
        if fc not in RANDOM_FACTOR_COUNTS:
            p.error(
                f"--factor-count must be one of {sorted(RANDOM_FACTOR_COUNTS)} for strategy={strategy} "
                f"(got {fc})"
            )
    elif strategy == "stairvss":
        if fc not in STAIR_FACTOR_COUNTS:
            p.error(
                f"--factor-count must be one of {sorted(STAIR_FACTOR_COUNTS)} for strategy={strategy} "
                f"(got {fc})"
            )
    elif strategy == "Lrandom":
        if fc not in LRANDOM_FACTOR_COUNTS:
            p.error(
                f"--factor-count must be one of {sorted(LRANDOM_FACTOR_COUNTS)} for strategy=Lrandom "
                f"(got {fc})"
            )
    rng = random.Random(args.seed)
    experiment = "all_factor"
    record_dir = Path(args.record_base) / experiment / (args.record_segment or f"{strategy}_f{fc}")
    record_dir.mkdir(parents=True, exist_ok=True)

    support = build_support(strategy, rng, factor_count=int(args.factor_count))
    if not support:
        raise RuntimeError("internal: empty support")
    if strategy == "random":
        want = RANDOM_SUPPORT_LEN[fc]
        if len(support) != want:
            raise RuntimeError(f"internal: {strategy} support len {len(support)} != {want} for factor_count={fc}")
    pool = parse_size_pool_labels(args.size_pool)
    if pool != frozenset(range(len(ALL_FACTOR_SIZES))):
        support = remap_support_size_pool(support, pool, rng)
    if not support:
        raise RuntimeError("internal: empty support after size remap")

    pair_iter = _infinite_shuffled(support, rng)

    n = max(1, int(args.num_demos))
    existing = _existing_demo_indices(record_dir, strategy)
    success_count = len(existing)
    next_idx = (existing[-1] + 1) if existing else 0
    attempt_id = next_idx

    while success_count < n:
        idx_tuple = next(pair_iter)
        verb, color, shape, spatial, size_slug = _tuple_to_labels(idx_tuple)
        scene = pack_verb_object_color_options(
            size_label=size_slug,
            rng=rng,
            distractor_max_cli=int(args.distractor_max),
            num_distractors_cli=int(args.num_distractors),
            small_large_distractor_policy=str(args.small_large_distractor_policy),
            small_large_no_distractor_prob=float(args.small_large_no_distractor_prob),
            small_large_fixed_num_distractors=int(args.small_large_fixed_num_distractors),
        )
        n_d = int(scene["num_distractors"])
        traj_name = f"{strategy}_{next_idx:04d}_{verb}_{color}_{shape}_{spatial}_{size_slug}"
        run_kwargs = dict(
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
            obj_xy=_spatial_xy(spatial, rng),
            object_size_jiggle=0.0,
            distractor_max=max(int(args.distractor_max), n_d),
            target_size_scale=scene["target_size_scale"],
            distractor_size_scales=scene["distractor_size_scales"],
            num_distractors=scene["num_distractors"],
            distractor_match_target_geometry=scene["distractor_match_target_geometry"],
        )

        out = run_verb_color_shape_motion_planning(
            verb=verb,
            color=color,
            shape=shape,
            traj_name=traj_name,
            **run_kwargs,
        )
        attempt_id += 1
        if out is not None:
            success_count += 1
            next_idx += 1


if __name__ == "__main__":
    main()
