"""Primary pairs along index diagonal (k,k) + third-factor list indices.

The pipelines default to two third-factor values (``THIRD_INDICES=0,1`` in shell).
"""
from __future__ import annotations

from collection_strategy.lib.pairwise_factor_patterns import Experiment, primary_pair_from_indices
from collection_strategy.lib.training_vocab import (
    THIRD_COLORS_FOR_VERB_OBJECT,
    THIRD_OBJECTS_FOR_VERB_COLOR,
    THIRD_VERBS_FOR_COLOR_OBJECT,
    TRAINING_COLORS,
    TRAINING_SHAPES,
    TRAINING_VERBS,
)


def _third_full_list(experiment: Experiment) -> tuple[str, ...]:
    if experiment == "verb_color":
        return THIRD_OBJECTS_FOR_VERB_COLOR
    if experiment == "verb_object":
        return THIRD_COLORS_FOR_VERB_OBJECT
    if experiment == "color_object":
        return THIRD_VERBS_FOR_COLOR_OBJECT
    raise ValueError(experiment)


def parse_third_indices(spec: str) -> list[int]:
    """Parse comma/space-separated ints into non-negative indices (order preserved, duplicates dropped)."""
    out: list[int] = []
    seen: set[int] = set()
    for part in spec.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        i = int(part, 10)
        if i < 0:
            raise ValueError(f"third index must be >= 0, got {i}")
        if i not in seen:
            seen.add(i)
            out.append(i)
    if not out:
        raise ValueError("empty --third-indices")
    return out


def third_pool_for_indices(experiment: Experiment, indices: list[int]) -> tuple[str, ...]:
    full = _third_full_list(experiment)
    pool: list[str] = []
    for i in indices:
        if i >= len(full):
            raise ValueError(
                f"third index {i} out of range for {experiment} (len={len(full)}): {full!r}"
            )
        pool.append(full[i])
    return tuple(pool)


def _diagonal_extent(experiment: Experiment) -> int:
    """How many diagonal cells (k,k) exist for this experiment (min of the two primary axis lengths)."""
    if experiment == "verb_color":
        return min(len(TRAINING_COLORS), len(TRAINING_VERBS))
    if experiment == "verb_object":
        return min(len(TRAINING_SHAPES), len(TRAINING_VERBS))
    if experiment == "color_object":
        return min(len(TRAINING_COLORS), len(TRAINING_SHAPES))
    raise ValueError(experiment)


def fixed_diagonal_primary_pairs(experiment: Experiment) -> list[tuple[str, ...]]:
    """Primary pairs ``(k,k)`` for ``k = 0 .. n-1`` (verb_color: color/verb; verb_object: shape/verb; color_object: color/shape)."""
    n = _diagonal_extent(experiment)
    return [primary_pair_from_indices(experiment, (k, k)) for k in range(n)]
