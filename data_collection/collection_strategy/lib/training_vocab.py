"""Training vocabulary for pairwise collection (see collection_strategy/README.md)."""

from __future__ import annotations

# 6 verbs × 6 colors × 6 shapes in training scope
TRAINING_VERBS: tuple[str, ...] = (
    "lift",
    "grasp",
    "push",
    "pull",
    "rotate",
    "slide",
)

TRAINING_SHAPES: tuple[str, ...] = (
    "cube",
    "sphere",
    "cup",
    "car",
    "pyramid",
    "star",
)

TRAINING_COLORS: tuple[str, ...] = (
    "red",
    "yellow",
    "blue",
    "orange",
    "green",
    "black",
)

# Third controlled variable: first N entries (N = --third-pool-size, default 3) when sweeping the other two factors
THIRD_OBJECTS_FOR_VERB_COLOR: tuple[str, ...] = ("cube", "sphere", "cup")
THIRD_COLORS_FOR_VERB_OBJECT: tuple[str, ...] = ("red", "yellow", "blue")
THIRD_VERBS_FOR_COLOR_OBJECT: tuple[str, ...] = ("lift", "grasp", "push")
