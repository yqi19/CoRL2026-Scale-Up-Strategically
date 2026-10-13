#!/usr/bin/env python3
"""
Collect attribute-focused demos:
- verb_size: verb × size-label (fixed color/shape)
- verb_spatial: verb × spatial-label (fixed color/shape/size)
- color_size: color × size-label (fixed verb=lift, shape=cube)
- spatial_size: spatial × size-label (fixed verb=lift, color=red, shape=cube)

Sampling strategies and factor-grid logic reuse the existing pairwise index patterns.
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
    random_support_indices,
)
from collection_strategy.lib.pairwise_strategies import Strategy
from scripts.run_verb_color_shape_motion_planning import run_verb_color_shape_motion_planning

AttributeExperiment = str
VERB_SET: tuple[str, ...] = ("lift", "grasp", "push", "pull", "rotate", "slide")
# Size axis for ``verb_size``: same six labels and **order** as
# ``conflict_experiment/collect_conflict_attribute.py`` ``SIZES``.
# - absolute: ``small``, ``large``
# - two-object relative: ``smaller``, ``larger``
# - three-object relative: ``smallest``, ``largest``
VERB_SIZE_STEMS_BY_GRID_I: tuple[str, ...] = (
    "small",
    "large",
    "smaller",
    "larger",
    "smallest",
    "largest",
)
# All stems that may appear in filenames for new verb_size collection.
SIZE_SET: tuple[str, ...] = ("small", "large", "smaller", "larger", "smallest", "largest")
# verb_spatial uses 5 spatial relations: left / right / middle / front / behind.
# The pairwise 6x6 index patterns are reused by mapping the 6th slot to "middle" as an alias.
SPATIAL_SET: tuple[str, ...] = ("left", "right", "middle", "front", "behind", "middle")
# color axis for color_size (6 colors, matching conflict_experiment COLORS order).
COLOR_SET: tuple[str, ...] = ("red", "yellow", "blue", "orange", "green", "black")


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


def _primary_pair(experiment: AttributeExperiment, ij: tuple[int, int]) -> tuple[str, str]:
    i, j = ij
    if experiment == "verb_size":
        return VERB_SET[j], VERB_SIZE_STEMS_BY_GRID_I[i]
    if experiment == "verb_spatial":
        return VERB_SET[j], SPATIAL_SET[i]
    if experiment == "color_size":
        # factor1=color (row i), factor2=size (col j)
        return COLOR_SET[i], VERB_SIZE_STEMS_BY_GRID_I[j]
    if experiment == "spatial_size":
        # factor1=spatial (row i), factor2=size (col j)
        return SPATIAL_SET[i], VERB_SIZE_STEMS_BY_GRID_I[j]
    raise ValueError(experiment)


def _build_support(
    experiment: AttributeExperiment,
    strategy: Strategy,
    factor_count: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    if strategy == "random":
        idxs = random_support_indices(factor_count, rng)
        return [_primary_pair(experiment, ij) for ij in idxs]
    if strategy in ("stair", "Lrandom"):
        # Reuse the pairwise index patterns, then remap indices to this experiment's factor labels.
        from collection_strategy.lib.pairwise_factor_patterns import (
            LRANDOM_6,
            LRANDOM_12,
            Lrandom_18_index_support,
            STAIR_6,
            STAIR_12,
            STAIR_18,
        )

        if strategy == "stair":
            idxs = STAIR_6 if factor_count == 6 else STAIR_12 if factor_count == 12 else STAIR_18
        elif strategy == "Lrandom":
            if factor_count == 6:
                idxs = LRANDOM_6
            elif factor_count == 12:
                idxs = LRANDOM_12
            else:
                idxs = Lrandom_18_index_support(rng)
        else:
            raise ValueError(strategy)

        return [_primary_pair(experiment, ij) for ij in idxs]
    raise ValueError(strategy)


def _sample_small_large_distractors(
    rng: random.Random, size_label: str, distractor_cap: int
) -> tuple[float, list[float] | None, int]:
    """Absolute ``small`` / ``large``: always no distractors (target-only). ``distractor_cap`` is ignored."""
    del rng, distractor_cap
    if size_label == "small":
        return 0.72, None, 0
    if size_label == "large":
        return 1.34, None, 0
    raise ValueError(size_label)


def _size_controls(size_label: str) -> tuple[float, list[float], int]:
    # Relative / contrastive size modes (fixed layout, always with distractors). ``small`` / ``large`` use
    # ``_sample_small_large_distractors`` instead (no distractors).
    if size_label == "smaller":
        return 0.82, [1.08], 1
    if size_label == "larger":
        return 1.18, [0.92], 1
    if size_label == "smallest":
        return 0.78, [1.00, 1.24], 2
    if size_label == "largest":
        return 1.26, [1.00, 0.80], 2
    raise ValueError(size_label)


def _spatial_xy(spatial: str, rng: random.Random) -> list[float]:
    # Workspace anchors with small jitter.
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


def main() -> None:
    p = argparse.ArgumentParser(description="Pairwise attribute collection (verb_size / verb_spatial / color_size / spatial_size)")
    p.add_argument("--experiment", type=str, required=True, choices=("verb_size", "verb_spatial", "color_size", "spatial_size"))
    p.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=("random", "stair", "Lrandom"),
    )
    p.add_argument("--factor-count", type=int, default=6, choices=(6, 12, 18))
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
    p.add_argument("--obj-xy-range", type=float, nargs=4, default=None)
    p.add_argument("--num-distractors", type=int, default=2, metavar="N")
    p.add_argument("--distractor-max", type=int, default=2, metavar="N")
    args = p.parse_args()

    experiment = args.experiment
    strategy: Strategy = args.strategy  # type: ignore[assignment]
    factor_count = int(args.factor_count)

    rng = random.Random(args.seed)
    record_dir = Path(args.record_base) / experiment / (args.record_segment or strategy)
    record_dir.mkdir(parents=True, exist_ok=True)

    support = _build_support(experiment, strategy, factor_count, rng)
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

    while success_count < n:
        fac_a, fac_b = pick_pair()
        traj_name: str
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
            distractor_max=max(0, min(3, int(args.distractor_max))),
            num_distractors=max(0, min(3, int(args.num_distractors))),
        )
        if experiment == "verb_size":
            verb, attr = fac_a, fac_b
            color = "red"
            shape = "cube"
            # Upper bound for how many distractors may appear (env allows up to 3).
            cap_dm = max(0, min(3, int(args.distractor_max), int(args.num_distractors)))
            if attr in ("small", "large"):
                t_scale, d_opt, n_d = _sample_small_large_distractors(rng, attr, cap_dm)
                run_kwargs["target_size_scale"] = t_scale
                run_kwargs["distractor_size_scales"] = d_opt
                run_kwargs["num_distractors"] = n_d
            else:
                t_scale, d_scales, n_d = _size_controls(attr)
                run_kwargs["target_size_scale"] = t_scale
                run_kwargs["distractor_size_scales"] = d_scales or None
                run_kwargs["num_distractors"] = n_d
            run_kwargs["distractor_max"] = max(int(args.distractor_max), n_d)
            run_kwargs["object_size_jiggle"] = 0.0
            traj_name = f"{strategy}_{next_idx:04d}_{verb}_{attr}_{shape}"
        elif experiment == "verb_spatial":
            verb, attr = fac_a, fac_b
            color = "red"
            shape = "cube"
            run_kwargs["obj_xy"] = _spatial_xy(attr, rng)
            run_kwargs["object_size_jiggle"] = 0.0
            run_kwargs["target_size_scale"] = 1.0
            traj_name = f"{strategy}_{next_idx:04d}_{verb}_{attr}_{shape}"
        elif experiment == "color_size":
            color, size_label = fac_a, fac_b
            verb = "lift"
            shape = "cube"
            cap_dm = max(0, min(3, int(args.distractor_max), int(args.num_distractors)))
            if size_label in ("small", "large"):
                t_scale, d_opt, n_d = _sample_small_large_distractors(rng, size_label, cap_dm)
                run_kwargs["target_size_scale"] = t_scale
                run_kwargs["distractor_size_scales"] = d_opt
                run_kwargs["num_distractors"] = n_d
            else:
                t_scale, d_scales, n_d = _size_controls(size_label)
                run_kwargs["target_size_scale"] = t_scale
                run_kwargs["distractor_size_scales"] = d_scales or None
                run_kwargs["num_distractors"] = n_d
            run_kwargs["distractor_max"] = max(int(args.distractor_max), n_d)
            run_kwargs["object_size_jiggle"] = 0.0
            attr_token = f"{color}-{size_label}"
            traj_name = f"{strategy}_{next_idx:04d}_{verb}_{attr_token}_{shape}"
        else:  # spatial_size
            spatial, size_label = fac_a, fac_b
            verb = "lift"
            color = "red"
            shape = "cube"
            cap_dm = max(0, min(3, int(args.distractor_max), int(args.num_distractors)))
            if size_label in ("small", "large"):
                t_scale, d_opt, n_d = _sample_small_large_distractors(rng, size_label, cap_dm)
                run_kwargs["target_size_scale"] = t_scale
                run_kwargs["distractor_size_scales"] = d_opt
                run_kwargs["num_distractors"] = n_d
            else:
                t_scale, d_scales, n_d = _size_controls(size_label)
                run_kwargs["target_size_scale"] = t_scale
                run_kwargs["distractor_size_scales"] = d_scales or None
                run_kwargs["num_distractors"] = n_d
            run_kwargs["distractor_max"] = max(int(args.distractor_max), n_d)
            run_kwargs["object_size_jiggle"] = 0.0
            run_kwargs["obj_xy"] = _spatial_xy(spatial, rng)
            attr_token = f"{spatial}-{size_label}"
            traj_name = f"{strategy}_{next_idx:04d}_{verb}_{attr_token}_{shape}"

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
