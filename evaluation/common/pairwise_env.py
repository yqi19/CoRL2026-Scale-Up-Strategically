#!/usr/bin/env python3
"""Unified ManiSkill interface for the four original pairwise evaluation grids.

This module contains no model-specific code.  A policy evaluator can call
``build_cell`` and receive a registered ``VerbObjectColor-v1`` environment,
the reset options, and the natural-language instruction.

Grids:
  verb_color:  verb(6) x color(6), shape sampled from cube/sphere/cup
  color_object: color(6) x shape(6), verb sampled from lift/grasp/push
  verb_object: verb(6) x shape(6), color sampled from red/yellow/blue
  verb_size: verb(6) x size(6), fixed red cube
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import random
import sys
from typing import Any


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import setup_paths  # noqa: E402

setup_paths()

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402,F401 - registers VerbObjectColor-v1

from collection_strategy.lib.pairwise_task_language import VERB_TO_EN  # noqa: E402
from collection_strategy.lib.training_vocab import (  # noqa: E402
    THIRD_COLORS_FOR_VERB_OBJECT,
    THIRD_OBJECTS_FOR_VERB_COLOR,
    THIRD_VERBS_FOR_COLOR_OBJECT,
    TRAINING_COLORS,
    TRAINING_SHAPES,
    TRAINING_VERBS,
)


EXPERIMENTS = ("verb_color", "color_object", "verb_object", "verb_size")
SIZES = ("small", "large", "smaller", "larger", "smallest", "largest")
COLOR_TO_ID = {color: i for i, color in enumerate(TRAINING_COLORS)}
SIZE_CONFIG = {
    "small": (0.72, [], 0),
    "large": (1.34, [], 0),
    "smaller": (0.82, [1.08], 1),
    "larger": (1.18, [0.92], 1),
    "smallest": (0.78, [1.00, 1.24], 2),
    "largest": (1.26, [1.00, 0.80], 2),
}


@dataclasses.dataclass(frozen=True)
class Cell:
    experiment: str
    factor_a: str
    factor_b: str
    verb: str
    color: str
    shape: str
    instruction: str
    make_kwargs: dict[str, Any]
    reset_options: dict[str, Any]


def factor_values(experiment: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the two ordered axes of an evaluation grid."""
    if experiment == "verb_color":
        return TRAINING_VERBS, TRAINING_COLORS
    if experiment == "color_object":
        return TRAINING_COLORS, TRAINING_SHAPES
    if experiment == "verb_object":
        return TRAINING_VERBS, TRAINING_SHAPES
    if experiment == "verb_size":
        return TRAINING_VERBS, SIZES
    raise ValueError(f"unknown experiment {experiment!r}; choose from {EXPERIMENTS}")


def build_cell(
    experiment: str,
    factor_a: str,
    factor_b: str,
    *,
    seed: int = 42,
    third_pool_size: int = 2,
    sim_backend: str = "cpu",
    max_episode_steps: int = 200,
    task_difficulty: float = 1.0,
) -> Cell:
    """Materialize one grid cell using the original evaluation conventions."""
    rng = random.Random(seed)
    pool_n = max(1, int(third_pool_size))
    reset_options: dict[str, Any] = {}

    if experiment == "verb_color":
        verb, color = factor_a, factor_b
        shape = rng.choice(THIRD_OBJECTS_FOR_VERB_COLOR[:pool_n])
    elif experiment == "color_object":
        color, shape = factor_a, factor_b
        verb = rng.choice(THIRD_VERBS_FOR_COLOR_OBJECT[:pool_n])
    elif experiment == "verb_object":
        verb, shape = factor_a, factor_b
        color = rng.choice(THIRD_COLORS_FOR_VERB_OBJECT[:pool_n])
    elif experiment == "verb_size":
        verb, size = factor_a, factor_b
        color, shape = "red", "cube"
        target_scale, distractor_scales, num_distractors = SIZE_CONFIG[size]
        reset_options = {
            "num_distractors": num_distractors,
            "target_size_scale": target_scale,
            "distractor_size_scales": distractor_scales,
        }
    else:
        raise ValueError(f"unknown experiment {experiment!r}; choose from {EXPERIMENTS}")

    axes = factor_values(experiment)
    if factor_a not in axes[0] or factor_b not in axes[1]:
        raise ValueError(
            f"invalid {experiment} cell ({factor_a!r}, {factor_b!r}); axes={axes}"
        )

    instruction = VERB_TO_EN[verb].format(color=color, shape=shape)
    if experiment == "verb_size":
        instruction = f"{verb.capitalize()} the {factor_b} {color} {shape}."

    distractor_max = max(2, int(reset_options.get("num_distractors", 0)))
    make_kwargs = {
        "obs_mode": "rgb",
        "control_mode": "pd_joint_pos",
        "sim_backend": sim_backend,
        "render_backend": sim_backend,
        "max_episode_steps": max_episode_steps,
        "task_difficulty": task_difficulty,
        "verb": verb,
        "object_shape": shape,
        "object_color_id": COLOR_TO_ID[color],
        "distractor_max": distractor_max,
    }
    if experiment == "verb_size":
        make_kwargs.update(
            object_size_jiggle=0.0,
            target_size_scale=target_scale,
            distractor_size_scales=distractor_scales,
            distractor_specs=[("cube", COLOR_TO_ID[color])] * num_distractors
            + [None] * (3 - num_distractors),
        )
    return Cell(
        experiment=experiment,
        factor_a=factor_a,
        factor_b=factor_b,
        verb=verb,
        color=color,
        shape=shape,
        instruction=instruction,
        make_kwargs=make_kwargs,
        reset_options=reset_options,
    )


def make_env(cell: Cell) -> gym.Env:
    """Construct the registered environment for a materialized cell."""
    return gym.make("VerbObjectColor-v1", **cell.make_kwargs)


def iter_grid(experiment: str, **kwargs: Any):
    """Yield all 36 cells in stable row-major order."""
    axis_a, axis_b = factor_values(experiment)
    for a in axis_a:
        for b in axis_b:
            yield build_cell(experiment, a, b, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--third-pool-size", type=int, default=2)
    parser.add_argument("--smoke-reset", action="store_true")
    parser.add_argument("--sim-backend", default="cpu", choices=("cpu", "gpu"))
    args = parser.parse_args()

    cells = list(
        iter_grid(
            args.experiment,
            seed=args.seed,
            third_pool_size=args.third_pool_size,
            sim_backend=args.sim_backend,
        )
    )
    print(f"{args.experiment}: {len(cells)} cells")
    for index, cell in enumerate(cells, 1):
        print(index, cell.factor_a, cell.factor_b, "->", cell.instruction)

    if args.smoke_reset:
        cell = cells[0]
        env = make_env(cell)
        try:
            obs, info = env.reset(seed=args.seed, options=cell.reset_options)
            print("SMOKE_RESET_OK", sorted(obs), sorted(info))
        finally:
            env.close()


if __name__ == "__main__":
    main()
