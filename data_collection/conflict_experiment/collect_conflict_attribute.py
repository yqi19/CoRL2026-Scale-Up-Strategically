#!/usr/bin/env python3
"""
Conflict attribute demos: primary diagonal pairs per ``--experiment``; third factor from
``--third-indices`` into ``OBJECTS`` (shape in text) or ``VERBS`` (verb in text), default ``0,1``.

- Shapes as third pool: ``verb_size``, ``verb_spatial``, ``spatial_size`` (``spatial_size`` also fixes verb=lift, color=red).
- Verbs as third pool: ``object_spatial``, ``spatial_object``, ``size_object``, ``color_spatial``, ``color_size`` (latter two: shape cube, color from diagonal).
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_verb_color_shape_motion_planning import run_verb_color_shape_motion_planning

VERBS: tuple[str, ...] = ("lift", "grasp", "push", "pull", "rotate", "slide")
SIZES: tuple[str, ...] = ("small", "large", "smaller", "larger", "smallest", "largest")
OBJECTS: tuple[str, ...] = ("cube", "sphere", "cup", "car", "pyramid", "star")
COLORS: tuple[str, ...] = ("red", "yellow", "blue", "orange", "green", "black")
SPATIALS: tuple[str, ...] = ("left", "right", "middle", "front", "behind")
_TRAJ_PREFIX = "conflict"


def _existing_demo_indices(record_dir: Path) -> list[int]:
    pat = re.compile(rf"^{re.escape(_TRAJ_PREFIX)}_(\d+)_")
    idxs: set[int] = set()
    for pth in record_dir.rglob("*.h5"):
        stem = pth.stem
        if stem.endswith("_rgb"):
            continue
        m = pat.match(stem)
        if m:
            idxs.add(int(m.group(1)))
    return sorted(idxs)


def _diag_pairs(experiment: str) -> list[tuple[str, str]]:
    if experiment == "verb_size":
        n = min(len(SIZES), len(VERBS))
        return [(VERBS[k], SIZES[k]) for k in range(n)]
    if experiment == "verb_spatial":
        n = min(len(SPATIALS), len(VERBS))
        return [(VERBS[k], SPATIALS[k]) for k in range(n)]
    if experiment == "object_spatial":
        n = min(len(SPATIALS), len(OBJECTS))
        return [(OBJECTS[k], SPATIALS[k]) for k in range(n)]
    if experiment == "color_spatial":
        n = min(len(SPATIALS), len(COLORS))
        return [(COLORS[k], SPATIALS[k]) for k in range(n)]
    if experiment == "color_size":
        n = min(len(SIZES), len(COLORS))
        return [(COLORS[k], SIZES[k]) for k in range(n)]
    if experiment == "spatial_size":
        # Spatial vocabulary has 5 anchors; size has 6 labels — diagonal length is 5.
        n = min(len(SPATIALS), len(SIZES))
        return [(SPATIALS[k], SIZES[k]) for k in range(n)]
    if experiment == "spatial_object":
        n = min(len(SPATIALS), len(OBJECTS))
        return [(SPATIALS[k], OBJECTS[k]) for k in range(n)]
    if experiment == "size_object":
        n = min(len(SIZES), len(OBJECTS))
        return [(SIZES[k], OBJECTS[k]) for k in range(n)]
    raise ValueError(experiment)


def _parse_third_indices(spec: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for part in spec.replace(",", " ").split():
        idx = int(part, 10)
        if idx < 0:
            raise ValueError(f"third index must be >= 0, got {idx}")
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    if len(out) < 2:
        raise ValueError("--third-indices must contain at least 2 distinct indices")
    return out


def _third_full_list(experiment: str) -> tuple[str, ...]:
    if experiment in ("verb_size", "verb_spatial", "spatial_size"):
        # Third factor must appear in instruction; use object shape.
        return OBJECTS
    if experiment in ("object_spatial", "spatial_object", "size_object", "color_spatial", "color_size"):
        # Third factor appears as action verb in instruction.
        return VERBS
    raise ValueError(experiment)


def _third_pool(experiment: str, indices: list[int]) -> tuple[str, ...]:
    full = _third_full_list(experiment)
    pool: list[str] = []
    for idx in indices:
        if idx >= len(full):
            raise ValueError(f"third index {idx} out of range for {experiment} (len={len(full)})")
        pool.append(full[idx])
    return tuple(pool)


def _size_controls(size_label: str) -> tuple[float, list[float], int]:
    if size_label == "small":
        return 0.72, [], 0
    if size_label == "large":
        return 1.34, [], 0
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
    anchors = {
        "left": (-0.10, 0.00),
        "right": (0.10, 0.00),
        "middle": (0.00, 0.00),
        "front": (0.00, 0.10),
        "behind": (0.00, -0.10),
    }
    ax, ay = anchors[spatial]
    return [ax + rng.uniform(-0.012, 0.012), ay + rng.uniform(-0.012, 0.012)]


def _infinite_shuffled(base_pairs: list[tuple[str, str]], rng: random.Random):
    while True:
        cyc = list(base_pairs)
        rng.shuffle(cyc)
        yield from cyc


def main() -> None:
    p = argparse.ArgumentParser(
        description="Conflict collection for verb_size / verb_spatial / object_spatial / color_spatial / color_size / spatial_size / spatial_object / size_object"
    )
    p.add_argument(
        "--experiment",
        required=True,
        choices=(
            "verb_size",
            "verb_spatial",
            "object_spatial",
            "color_spatial",
            "color_size",
            "spatial_size",
            "spatial_object",
            "size_object",
        ),
    )
    p.add_argument("--record-base", type=str, default="outputs/demo_gen")
    p.add_argument("--record-segment", type=str, default=None)
    p.add_argument(
        "--third-indices",
        type=str,
        default="0,1",
        help="Comma-separated indices for third-factor pool; must include >=2 distinct indices (default: 0,1).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-demos", type=int, default=300)
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
    args = p.parse_args()

    rng = random.Random(args.seed)
    third_pool = _third_pool(args.experiment, _parse_third_indices(args.third_indices))
    record_dir = Path(args.record_base) / args.experiment / (args.record_segment or _TRAJ_PREFIX)
    record_dir.mkdir(parents=True, exist_ok=True)

    support = _diag_pairs(args.experiment)
    pair_iter = _infinite_shuffled(support, rng)

    n = max(1, int(args.num_demos))
    existing = _existing_demo_indices(record_dir)
    success_count = len(existing)
    next_idx = (existing[-1] + 1) if existing else 0
    attempt_id = next_idx

    while success_count < n:
        pair_a, pair_b = next(pair_iter)
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
        )
        if args.experiment == "verb_size":
            verb, attr = pair_a, pair_b
            shape = rng.choice(third_pool)
            color = "red"
            t_scale, d_scales, n_d = _size_controls(attr)
            run_kwargs.update(
                target_size_scale=t_scale,
                distractor_size_scales=d_scales,
                num_distractors=n_d,
                distractor_max=max(2, n_d),
                object_size_jiggle=0.0,
            )
            traj_name = f"{_TRAJ_PREFIX}_{next_idx:04d}_{verb}_{attr}_{shape}"
        elif args.experiment == "verb_spatial":
            verb, attr = pair_a, pair_b
            shape = rng.choice(third_pool)
            color = "red"
            run_kwargs.update(
                obj_xy=_spatial_xy(attr, rng),
                target_size_scale=1.0,
                object_size_jiggle=0.0,
                num_distractors=0,
                distractor_max=0,
            )
            traj_name = f"{_TRAJ_PREFIX}_{next_idx:04d}_{verb}_{attr}_{shape}"
        elif args.experiment == "object_spatial":
            shape, attr = pair_a, pair_b
            verb = rng.choice(third_pool)
            color = "red"
            run_kwargs.update(
                obj_xy=_spatial_xy(attr, rng),
                target_size_scale=1.0,
                object_size_jiggle=0.0,
                num_distractors=0,
                distractor_max=0,
            )
            traj_name = f"{_TRAJ_PREFIX}_{next_idx:04d}_{verb}_{attr}_{shape}"
        elif args.experiment == "color_spatial":
            color, attr = pair_a, pair_b
            verb = rng.choice(third_pool)
            shape = "cube"
            run_kwargs.update(
                obj_xy=_spatial_xy(attr, rng),
                target_size_scale=1.0,
                object_size_jiggle=0.0,
                num_distractors=0,
                distractor_max=0,
            )
            # Encode color + spatial in the attr token to preserve 5-part stem format.
            attr_token = f"{color}-{attr}"
            traj_name = f"{_TRAJ_PREFIX}_{next_idx:04d}_{verb}_{attr_token}_{shape}"
        elif args.experiment == "color_size":
            color, size_label = pair_a, pair_b
            verb = rng.choice(third_pool)
            shape = "cube"
            t_scale, d_scales, n_d = _size_controls(size_label)
            run_kwargs.update(
                target_size_scale=t_scale,
                distractor_size_scales=d_scales,
                num_distractors=n_d,
                distractor_max=max(2, n_d),
                object_size_jiggle=0.0,
            )
            attr_token = f"{color}-{size_label}"
            traj_name = f"{_TRAJ_PREFIX}_{next_idx:04d}_{verb}_{attr_token}_{shape}"
        elif args.experiment == "spatial_size":
            spatial, size_label = pair_a, pair_b
            verb = "lift"
            shape = rng.choice(third_pool)
            color = "red"
            t_scale, d_scales, n_d = _size_controls(size_label)
            run_kwargs.update(
                obj_xy=_spatial_xy(spatial, rng),
                target_size_scale=t_scale,
                distractor_size_scales=d_scales,
                num_distractors=n_d,
                distractor_max=max(2, n_d),
                object_size_jiggle=0.0,
            )
            attr_token = f"{spatial}-{size_label}"
            traj_name = f"{_TRAJ_PREFIX}_{next_idx:04d}_{verb}_{attr_token}_{shape}"
        elif args.experiment == "spatial_object":
            spatial, shape = pair_a, pair_b
            verb = rng.choice(third_pool)
            color = "red"
            run_kwargs.update(
                obj_xy=_spatial_xy(spatial, rng),
                target_size_scale=1.0,
                object_size_jiggle=0.0,
                num_distractors=0,
                distractor_max=0,
            )
            traj_name = f"{_TRAJ_PREFIX}_{next_idx:04d}_{verb}_{spatial}_{shape}"
        elif args.experiment == "size_object":
            size_label, shape = pair_a, pair_b
            verb = rng.choice(third_pool)
            color = "red"
            t_scale, d_scales, n_d = _size_controls(size_label)
            run_kwargs.update(
                target_size_scale=t_scale,
                distractor_size_scales=d_scales,
                num_distractors=n_d,
                distractor_max=max(2, n_d),
                object_size_jiggle=0.0,
            )
            attr_token = f"{size_label}-{shape}"
            traj_name = f"{_TRAJ_PREFIX}_{next_idx:04d}_{verb}_{attr_token}_{shape}"
        else:
            raise AssertionError(f"unhandled experiment: {args.experiment}")

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
