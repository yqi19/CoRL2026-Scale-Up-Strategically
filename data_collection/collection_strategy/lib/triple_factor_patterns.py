"""
Index grids for ``verb_color_object``: full triples (verb_idx, color_idx, shape_idx).

Factor scaling (training vocab has 6 verbs / 6 colors / 6 shapes):

- **15** → use indices ``0..2`` per axis (3×3×3 cube)
- **20** → use indices ``0..3`` per axis (4×4×4 cube)
- **30** → use indices ``0..5`` per axis (full 6×6×6 cube)

``random`` keeps ``factor // 3`` triples, greedily spread in index space (L∞).

``stair``: **two** axis rays from the origin — verb ``(i,0,0)`` and object/shape ``(0,0,k)``
(no color-axis ray), plus the space diagonal ``(i, i, i)``, plus the 3D zigzag with
verb → color → shape priority per step.

``Lrandom`` uses the three coordinate rays ``(i,0,0)``, ``(0,j,0)``, ``(0,0,k)`` plus extras.
"""
from __future__ import annotations

import random

from collection_strategy.lib.training_vocab import (
    TRAINING_COLORS,
    TRAINING_SHAPES,
    TRAINING_VERBS,
)


def grid_side_for_factor(factor_count: int) -> int:
    if factor_count == 15:
        return 3
    if factor_count == 20:
        return 4
    if factor_count == 30:
        return 6
    raise ValueError(f"factor_count must be 15, 20, or 30, got {factor_count}")


def triple_from_indices(iv: int, ic: int, is_: int) -> tuple[str, str, str]:
    """Map (verb_idx, color_idx, shape_idx) → (verb, color, shape) strings."""
    return (
        TRAINING_VERBS[iv],
        TRAINING_COLORS[ic],
        TRAINING_SHAPES[is_],
    )


def _l_inf(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def _all_cells(side: int) -> list[tuple[int, int, int]]:
    return [(a, b, c) for a in range(side) for b in range(side) for c in range(side)]


def random_support_triples(factor_count: int, rng: random.Random) -> list[tuple[str, str, str]]:
    """``factor // 3`` triples, greedy max-min L∞ spread in index space."""
    side = grid_side_for_factor(factor_count)
    n_pick = max(1, factor_count // 3)
    cells = _all_cells(side)
    if n_pick >= len(cells):
        out = cells[:]
        rng.shuffle(out)
        return [triple_from_indices(*t) for t in out]

    first = rng.choice(cells)
    chosen: list[tuple[int, int, int]] = [first]
    remaining = [c for c in cells if c != first]

    while len(chosen) < n_pick and remaining:

        def score(c: tuple[int, int, int]) -> tuple[int, float]:
            dmin = min(_l_inf(c, x) for x in chosen)
            return dmin, rng.random()

        best = max(remaining, key=score)
        chosen.append(best)
        remaining.remove(best)

    return [triple_from_indices(*t) for t in chosen]


def _lrandom_three_axis_rays(side: int) -> set[tuple[int, int, int]]:
    """Three edges through origin: verb ``(i,0,0)``, color ``(0,j,0)``, shape ``(0,0,k)``."""
    s: set[tuple[int, int, int]] = set()
    for i in range(side):
        s.add((i, 0, 0))
    for j in range(side):
        s.add((0, j, 0))
    for k in range(side):
        s.add((0, 0, k))
    return s


def _verb_object_axis_rays(side: int) -> set[tuple[int, int, int]]:
    """``stair`` only: verb ray ``(i,0,0)`` and shape ray ``(0,0,k)`` (color index fixed at 0)."""
    s: set[tuple[int, int, int]] = set()
    for i in range(side):
        s.add((i, 0, 0))
    for k in range(side):
        s.add((0, 0, k))
    return s


def _cube_space_diagonal(side: int) -> set[tuple[int, int, int]]:
    """Body diagonal ``(i, i, i)`` in index space ``(verb, color, shape)``."""
    return {(i, i, i) for i in range(side)}


def _cube_zigzag_path(side: int, axis_cycle: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """From ``(0,0,0)`` to ``(S-1,S-1,S-1)``, +1 on one axis per step.

    At step ``t``, try to increment ``axis_cycle[t % 3]`` first; if that axis is
    already at ``S-1``, try the next axis in cyclic order until one moves. This
    yields e.g. ``(0,0,0)→(1,0,0)→(1,1,0)→(1,1,1)→…`` for ``axis_cycle=(0,1,2)``
    (verb, color, shape).
    """
    if side <= 0:
        return []
    pos = [0, 0, 0]
    out: list[tuple[int, int, int]] = [tuple(pos)]
    target = (side - 1, side - 1, side - 1)
    step = 0
    while tuple(pos) != target:
        moved = False
        for off in range(3):
            ax = axis_cycle[(step + off) % 3]
            if pos[ax] < side - 1:
                pos[ax] += 1
                out.append(tuple(pos))
                step += 1
                moved = True
                break
        if not moved:
            raise RuntimeError("cube zigzag stuck; side=%r" % (side,))
    return out


def lrandom_support_triples(factor_count: int, rng: random.Random) -> list[tuple[str, str, str]]:
    """Three axis rays through (0,0,0) + extra random cells (seed once per call)."""
    side = grid_side_for_factor(factor_count)
    base = _lrandom_three_axis_rays(side)
    if side == 3:
        n_extra = 1
    elif side == 4:
        n_extra = 2
    else:
        n_extra = 4

    all_cells = _all_cells(side)
    pool = [c for c in all_cells if c not in base]
    n_extra = min(n_extra, len(pool))
    if n_extra > 0:
        base.update(rng.sample(pool, n_extra))

    ordered = sorted(base)
    return [triple_from_indices(*t) for t in ordered]


def stair_support_triples(factor_count: int) -> list[tuple[str, str, str]]:
    """Verb + object rays, space diagonal, 3D zigzag (verb → color → shape per step)."""
    side = grid_side_for_factor(factor_count)
    s: set[tuple[int, int, int]] = set(_verb_object_axis_rays(side))
    s.update(_cube_space_diagonal(side))
    s.update(_cube_zigzag_path(side, (0, 1, 2)))
    ordered = sorted(s)
    return [triple_from_indices(*t) for t in ordered]


def build_support(
    strategy: str,
    factor_count: int,
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    if strategy == "random":
        return random_support_triples(factor_count, rng)
    if strategy == "Lrandom":
        return lrandom_support_triples(factor_count, rng)
    if strategy == "stair":
        return stair_support_triples(factor_count)
    raise ValueError(strategy)
