"""Strategy names and helpers for pairwise collection (factor grids 6 / 12 / 18)."""
from __future__ import annotations

import random
from typing import Literal

from collection_strategy.lib.pairwise_factor_patterns import (
    Experiment,
    structured_support,
)

Strategy = Literal[
    "random",
    "stair",
    "Lrandom",
]


def count_support_pairs(strategy: Strategy, experiment: Experiment, factor_count: int) -> int:
    """Size of discrete primary support (for logging)."""
    if strategy == "random":
        if factor_count == 6:
            return 3
        if factor_count == 12:
            return 6
        if factor_count == 18:
            return 9
        raise ValueError(factor_count)
    need_rng = strategy == "Lrandom" and factor_count == 18
    s = structured_support(
        experiment,
        strategy,
        factor_count,
        random.Random(0) if need_rng else None,
    )
    return len(s)
