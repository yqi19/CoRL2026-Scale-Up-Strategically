"""Index grids and fixed (factor1, factor2) pair lists for pairwise collection.

Convention (indices into TRAINING_* tuples):
- verb_color:   (factor1, factor2) = (color_idx, verb_idx)   → pair strings (verb, color)
- verb_object:  (factor1, factor2) = (shape_idx, verb_idx)   → pair strings (verb, shape)
- color_object: (factor1, factor2) = (color_idx, shape_idx) → pair strings (color, shape)
"""
from __future__ import annotations

import random
from typing import Literal

from collection_strategy.lib.training_vocab import (
    TRAINING_COLORS,
    TRAINING_SHAPES,
    TRAINING_VERBS,
)

Experiment = Literal["verb_color", "verb_object", "color_object"]

# ---------------------------------------------------------------------------
# Fixed patterns (index space)
# ---------------------------------------------------------------------------

# factor_count == 6  (stair uses indices 0..2; random uses 0..1)
STAIR_6: list[tuple[int, int]] = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]

# factor_count == 12  (indices 0..3)
STAIR_12: list[tuple[int, int]] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 1),
    (2, 1),
    (2, 2),
    (3, 2),
    (3, 3),
]


def _dedupe_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# Lrandom — custom index lists; factor 18 adds 3 interior cells once per collect (see Lrandom_18_index_support)
LRANDOM_6: list[tuple[int, int]] = [(0, 0), (0, 1), (1, 0), (1, 1)]
LRANDOM_12: list[tuple[int, int]] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (2, 0),
    (3, 0),
    (1, 1),
    (1, 2),
]


def Lrandom_18_index_support(rng: random.Random) -> list[tuple[int, int]]:
    """Lrandom @ f18: row 0 + column 0 (deduped), then 3 random interior cells (i>=1, j>=1).

    Chosen once per call via ``rng.sample``; pass the run RNG so ``--seed`` fixes it for the whole collect.
    """
    side = 6
    out: list[tuple[int, int]] = []
    for j in range(side):
        out.append((0, j))
    for i in range(1, side):
        out.append((i, 0))
    interior = [(i, j) for i in range(1, side) for j in range(1, side)]
    out.extend(rng.sample(interior, 3))
    return out


def stair_18_pairs() -> list[tuple[int, int]]:
    """Row 0 full, then main diagonal (i,i) and parallel (i,i+1); indices 0..5."""
    s: list[tuple[int, int]] = []
    for k in range(6):
        s.append((0, k))
    for i in range(6):
        s.append((i, i))
    for i in range(5):
        s.append((i, i + 1))
    return _dedupe_pairs(s)


STAIR_18: list[tuple[int, int]] = stair_18_pairs()


def all_index_pairs(side: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(side) for j in range(side)]


def random_support_indices(factor_count: int, rng: random.Random) -> list[tuple[int, int]]:
    """random strategy: sample half the cells (3 / 6 / 9 pairs for 6 / 12 / 18)."""
    if factor_count == 6:
        n_pick, side = 3, 2
    elif factor_count == 12:
        n_pick, side = 6, 4
    elif factor_count == 18:
        n_pick, side = 9, 6
    else:
        raise ValueError(factor_count)
    universe = all_index_pairs(side)
    return rng.sample(universe, n_pick)


def primary_pair_from_indices(
    experiment: Experiment, ij: tuple[int, int]
) -> tuple[str, ...]:
    """Map (factor1_idx, factor2_idx) to the primary pair tuple used by collect_pairwise."""
    i, j = ij
    if experiment == "verb_color":
        return (TRAINING_VERBS[j], TRAINING_COLORS[i])
    if experiment == "verb_object":
        return (TRAINING_VERBS[j], TRAINING_SHAPES[i])
    if experiment == "color_object":
        return (TRAINING_COLORS[i], TRAINING_SHAPES[j])
    raise ValueError(experiment)


# ---------------------------------------------------------------------------
# Non-square grid helpers (spatial_object: 5 spatial rows x 6 shape columns)
# ---------------------------------------------------------------------------

def spatial_object_stair(factor_count: int) -> list[tuple[int, int]]:
    """Stair (col-first / shape axis) for spatial_object 5×6 grid.

    - f12: cols 0,1 (10 cells) + (0,2),(1,2) → 12 cells
    - f20: cols 0,1,2,3 (20 cells)
    - f30: all 30 cells
    """
    if factor_count == 12:
        return (
            [(i, 0) for i in range(5)] +
            [(i, 1) for i in range(5)] +
            [(0, 2), (1, 2)]
        )
    if factor_count == 20:
        return [(i, j) for i in range(5) for j in range(4)]
    if factor_count == 30:
        return [(i, j) for i in range(5) for j in range(6)]
    raise ValueError(f"spatial_object_stair: unsupported factor_count={factor_count}")


def random_grid_indices(n_rows: int, n_cols: int, n_pick: int, rng: random.Random) -> list[tuple[int, int]]:
    """Sample ``n_pick`` unique cells uniformly from an ``n_rows × n_cols`` grid.

    If ``n_pick >= n_rows * n_cols``, all cells are returned (shuffled).
    """
    all_cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    if n_pick >= len(all_cells):
        return all_cells
    return rng.sample(all_cells, n_pick)


def lrandom_grid_indices(n_rows: int, n_cols: int, n_total: int, rng: random.Random) -> list[tuple[int, int]]:
    """Lrandom for an ``n_rows × n_cols`` grid.

    Base L = row 0 (n_cols cells) + col 0 minus corner (n_rows-1 cells) = n_cols + n_rows - 1 cells.
    Then fill with random interior cells to reach ``n_total``.

    If n_total <= base L size, returns first n_total cells of the L.
    If n_total >= all cells, returns all cells.
    """
    l_base: list[tuple[int, int]] = []
    for j in range(n_cols):
        l_base.append((0, j))
    for i in range(1, n_rows):
        l_base.append((i, 0))

    if n_total <= len(l_base):
        return l_base[:n_total]

    interior = [(i, j) for i in range(1, n_rows) for j in range(1, n_cols)]
    n_extra = n_total - len(l_base)
    if n_extra >= len(interior):
        return l_base + interior
    return l_base + rng.sample(interior, n_extra)


def structured_support(
    experiment: Experiment,
    strategy: str,
    factor_count: int,
    rng: random.Random | None = None,
) -> list[tuple[str, ...]]:
    """Discrete primary pairs for stair / Lrandom.

    ``rng`` is required for ``Lrandom`` with ``factor_count == 18``.
    """
    if factor_count == 6:
        if strategy == "stair":
            idxs = STAIR_6
        elif strategy == "Lrandom":
            idxs = LRANDOM_6
        else:
            raise ValueError(strategy)
    elif factor_count == 12:
        if strategy == "stair":
            idxs = STAIR_12
        elif strategy == "Lrandom":
            idxs = LRANDOM_12
        else:
            raise ValueError(strategy)
    elif factor_count == 18:
        if strategy == "stair":
            idxs = STAIR_18
        elif strategy == "Lrandom":
            if rng is None:
                raise ValueError("structured_support: Lrandom with factor_count=18 requires rng")
            idxs = Lrandom_18_index_support(rng)
        else:
            raise ValueError(strategy)
    else:
        raise ValueError(factor_count)
    return [primary_pair_from_indices(experiment, p) for p in idxs]
