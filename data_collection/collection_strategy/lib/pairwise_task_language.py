"""
Map motion-planning trajectory stems to natural-language task strings.

Expected stem pattern (from ``collect_pairwise``)::

    <strategy>_<dddd>_<verb>_<color>_<shape>[_rgb]

``strategy`` is one of: random, stair, Lrandom, conflict.
The triple (verb, color, shape) is the actual episode semantics; the language
instruction always describes that triple so mixed pairwise experiments still
get correct labels per file.
"""
from __future__ import annotations

from collection_strategy.lib.training_vocab import (
    TRAINING_COLORS,
    TRAINING_SHAPES,
    TRAINING_VERBS,
)

_STRATEGIES = frozenset(
    {"random", "stair", "Lrandom", "conflict"}
)
_VERB_SET = frozenset(TRAINING_VERBS)
_COLOR_SET = frozenset(TRAINING_COLORS)
_SHAPE_SET = frozenset(TRAINING_SHAPES)

# Instruction templates: must match executed control semantics (VerbObjectColor-v1).
VERB_TO_EN: dict[str, str] = {
    "lift": "Lift the {color} {shape}.",
    "grasp": "Grasp the {color} {shape}.",
    "push": "Push the {color} {shape}.",
    "pull": "Pull the {color} {shape}.",
    "rotate": "Rotate the {color} {shape}.",
    "slide": "Slide the {color} {shape}.",
}


def parse_pairwise_traj_stem(stem: str) -> tuple[str, str, str, str, str]:
    """
    Parse ``stem`` (with or without trailing ``_rgb``) into
    (strategy, demo_index_str, verb, color, shape).

    Strategy may contain underscores (e.g. ``random_new``); we match the longest known prefix.
    """
    body = stem[:-4] if stem.endswith("_rgb") else stem
    for strategy in sorted(_STRATEGIES, key=len, reverse=True):
        prefix = f"{strategy}_"
        if not body.startswith(prefix):
            continue
        rest = body[len(prefix) :]
        parts = rest.split("_", 3)
        if len(parts) != 4:
            continue
        idx_s, verb, color, shape = parts
        if len(idx_s) != 4 or not idx_s.isdigit():
            continue
        if verb not in _VERB_SET:
            raise ValueError(f"unknown verb {verb!r}")
        if color not in _COLOR_SET:
            raise ValueError(f"unknown color {color!r}")
        if shape not in _SHAPE_SET:
            raise ValueError(f"unknown shape {shape!r}")
        return strategy, idx_s, verb, color, shape
    parts = body.split("_")
    if len(parts) != 5:
        raise ValueError(
            f"expected strategy_idx_verb_color_shape (strategy may be multi-token), got {body!r}"
        )
    strategy, idx_s, verb, color, shape = parts
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {sorted(_STRATEGIES)}")
    if len(idx_s) != 4 or not idx_s.isdigit():
        raise ValueError(f"expected 4-digit demo index, got {idx_s!r}")
    if verb not in _VERB_SET:
        raise ValueError(f"unknown verb {verb!r}")
    if color not in _COLOR_SET:
        raise ValueError(f"unknown color {color!r}")
    if shape not in _SHAPE_SET:
        raise ValueError(f"unknown shape {shape!r}")
    return strategy, idx_s, verb, color, shape


def traj_stem_to_task_label(stem: str) -> str:
    """H5 filename stem (e.g. ``random_0003_lift_red_cube_rgb``) -> English task string."""
    _strategy, _idx, verb, color, shape = parse_pairwise_traj_stem(stem)
    template = VERB_TO_EN.get(verb)
    if template is None:
        raise ValueError(f"no instruction template for verb {verb!r}")
    return template.format(color=color, shape=shape)
