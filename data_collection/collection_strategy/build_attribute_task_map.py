#!/usr/bin/env python3
"""Build task map JSON for verb_size / verb_spatial / … / all_factor replay rgb h5 files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

VERB_TO_EN: dict[str, str] = {
    "lift": "Lift",
    "grasp": "Grasp",
    "push": "Push",
    "pull": "Pull",
    "rotate": "Rotate",
    "slide": "Slide",
}
COLOR_SET = {"red", "yellow", "blue", "orange", "green", "black"}
# ``verb_size`` stems:
# - absolute: small / large
# - relative among 2 objects: smaller / larger
# - relative among 3 objects: smallest / largest
SIZE_SET = {"small", "large", "smaller", "larger", "smallest", "largest"}
# Map alternative size tokens in stems to the current size slug.
# Do not include current ``SIZE_SET`` keys here except aliases that never appear as new stems.
_SIZE_LEGACY_TO_SLUG: dict[str, str] = {
    "tiny": "small",
    "xsmall": "small",
    "xlarge": "large",
    "huge": "large",
    "huge_l": "large",
    "medium": "large",
}
# Stems use slug keys; language uses readable size adjectives before ``cube``.
SIZE_TO_EN_ADJ: dict[str, str] = {
    "small": "small",
    "large": "large",
    "smaller": "smaller",
    "larger": "larger",
    "smallest": "smallest",
    "largest": "largest",
}
OBJECT_SHAPE_SET = {"cube", "sphere", "cup", "car", "pyramid", "star"}
SPATIAL_TO_PHRASE = {
    "left": "on the left",
    "right": "on the right",
    "middle": "in the middle",
    "center": "in the middle",
    "front": "in front",
    # Parallel to left/right/front; bare "behind" after "the cube" is terse / ambiguous in English.
    "behind": "at the back",
}


def _label(experiment: str, stem: str) -> str:
    body = stem[:-4] if stem.endswith("_rgb") else stem
    parts = body.split("_")
    if experiment == "all_factor":
        # Parse from the right: last 5 tokens are always verb color shape spatial size_slug,
        # preceded by a 4-digit idx, preceded by the strategy name (may contain underscores).
        if len(parts) < 7:
            raise ValueError(f"bad all_factor stem (need at least 7 segments): {stem}")
        size_slug = parts[-1]
        spatial   = parts[-2]
        shape     = parts[-3]
        color     = parts[-4]
        verb      = parts[-5]
        _idx      = parts[-6]
        # _strategy = parts[:-6] (ignored, may have underscores)
        if verb not in VERB_TO_EN:
            raise ValueError(f"unknown verb in stem: {stem}")
        if color not in COLOR_SET:
            raise ValueError(f"unknown color in stem: {stem}")
        if shape not in OBJECT_SHAPE_SET:
            raise ValueError(f"unknown shape in stem: {stem}")
        phrase = SPATIAL_TO_PHRASE.get(spatial)
        if phrase is None:
            raise ValueError(f"unknown spatial label in stem: {stem}")
        if size_slug in SIZE_SET:
            size_out = size_slug
        elif size_slug in _SIZE_LEGACY_TO_SLUG:
            size_out = _SIZE_LEGACY_TO_SLUG[size_slug]
        else:
            raise ValueError(f"unknown size label in stem: {stem}")
        adj = SIZE_TO_EN_ADJ[size_out]
        return f"{VERB_TO_EN[verb]} the {adj} {color} {shape} {phrase}."
    if len(parts) != 5:
        raise ValueError(f"bad stem format: {stem}")
    _strategy, _idx, verb, attr, shape = parts
    if verb not in VERB_TO_EN:
        raise ValueError(f"unknown verb in stem: {stem}")
    if experiment == "verb_size":
        if shape != "cube":
            raise ValueError(f"expected fixed shape cube for verb_size, got {shape}")
        if attr in SIZE_SET:
            size_slug = attr
        elif attr in _SIZE_LEGACY_TO_SLUG:
            size_slug = _SIZE_LEGACY_TO_SLUG[attr]
        else:
            raise ValueError(f"unknown size label in stem: {stem}")
        adj = SIZE_TO_EN_ADJ[size_slug]
        return f"{VERB_TO_EN[verb]} the {adj} cube."
    if experiment in ("verb_spatial", "verb_spatial_new"):
        if shape != "cube":
            raise ValueError(f"expected fixed shape cube for verb_spatial, got {shape}")
        phrase = SPATIAL_TO_PHRASE.get(attr)
        if phrase is None:
            raise ValueError(f"unknown spatial label in stem: {stem}")
        return f"{VERB_TO_EN[verb]} the cube {phrase}."
    if experiment == "object_spatial":
        if verb not in VERB_TO_EN:
            raise ValueError(f"unknown verb in stem: {stem}")
        phrase = SPATIAL_TO_PHRASE.get(attr)
        if phrase is None:
            raise ValueError(f"unknown spatial label in stem: {stem}")
        return f"{VERB_TO_EN[verb]} the {shape} {phrase}."
    if experiment == "color_spatial":
        if verb not in VERB_TO_EN:
            raise ValueError(f"unknown verb in stem: {stem}")
        try:
            color, spatial = attr.split("-", 1)
        except ValueError as e:
            raise ValueError(f"bad color_spatial attr format in stem: {stem}") from e
        if color not in COLOR_SET:
            raise ValueError(f"unknown color in stem: {stem}")
        phrase = SPATIAL_TO_PHRASE.get(spatial)
        if phrase is None:
            raise ValueError(f"unknown spatial label in stem: {stem}")
        if shape != "cube":
            raise ValueError(f"expected fixed shape cube for color_spatial, got {shape}")
        return f"{VERB_TO_EN[verb]} the {color} cube {phrase}."
    if experiment == "color_size":
        if verb not in VERB_TO_EN:
            raise ValueError(f"unknown verb in stem: {stem}")
        try:
            color, size_token = attr.split("-", 1)
        except ValueError as e:
            raise ValueError(f"bad color_size attr format in stem: {stem}") from e
        if color not in COLOR_SET:
            raise ValueError(f"unknown color in stem: {stem}")
        if size_token in SIZE_SET:
            size_slug = size_token
        elif size_token in _SIZE_LEGACY_TO_SLUG:
            size_slug = _SIZE_LEGACY_TO_SLUG[size_token]
        else:
            raise ValueError(f"unknown size label in stem: {stem}")
        if shape != "cube":
            raise ValueError(f"expected fixed shape cube for color_size, got {shape}")
        adj = SIZE_TO_EN_ADJ[size_slug]
        return f"{VERB_TO_EN[verb]} the {adj} {color} cube."
    if experiment == "spatial_size":
        if shape != "cube":
            raise ValueError(f"expected fixed shape cube for spatial_size, got {shape}")
        try:
            spatial, size_token = attr.split("-", 1)
        except ValueError as e:
            raise ValueError(f"bad spatial_size attr format in stem: {stem}") from e
        phrase = SPATIAL_TO_PHRASE.get(spatial)
        if phrase is None:
            raise ValueError(f"unknown spatial label in stem: {stem}")
        if size_token in SIZE_SET:
            size_slug = size_token
        elif size_token in _SIZE_LEGACY_TO_SLUG:
            size_slug = _SIZE_LEGACY_TO_SLUG[size_token]
        else:
            raise ValueError(f"unknown size label in stem: {stem}")
        adj = SIZE_TO_EN_ADJ[size_slug]
        return f"{VERB_TO_EN[verb]} the {adj} cube {phrase}."
    if experiment == "spatial_object":
        phrase = SPATIAL_TO_PHRASE.get(attr)
        if phrase is None:
            raise ValueError(f"unknown spatial label in stem: {stem}")
        if shape not in OBJECT_SHAPE_SET:
            raise ValueError(f"unknown object shape in stem: {stem}")
        return f"{VERB_TO_EN[verb]} the {shape} {phrase}."
    if experiment == "size_object":
        try:
            size_token, shape_name = attr.split("-", 1)
        except ValueError as e:
            raise ValueError(f"bad size_object attr format in stem: {stem}") from e
        if shape_name != shape:
            raise ValueError(f"size_object stem shape mismatch attr vs tail: {stem}")
        if shape_name not in OBJECT_SHAPE_SET:
            raise ValueError(f"unknown object shape in stem: {stem}")
        if size_token in SIZE_SET:
            size_slug = size_token
        elif size_token in _SIZE_LEGACY_TO_SLUG:
            size_slug = _SIZE_LEGACY_TO_SLUG[size_token]
        else:
            raise ValueError(f"unknown size label in stem: {stem}")
        adj = SIZE_TO_EN_ADJ[size_slug]
        return f"{VERB_TO_EN[verb]} the {adj} {shape_name}."
    if experiment == "verb_object":
        # stem: strategy_idx_verb_shape_fixedcolor  (fixedcolor always 'red')
        # attr = obj_shape (e.g. 'cube'), shape = fixed_color ('red')
        obj_shape = attr  # parts[3]
        if obj_shape not in OBJECT_SHAPE_SET:
            raise ValueError(f"unknown shape in stem: {stem}")
        return f"{VERB_TO_EN[verb]} the red {obj_shape}."
    if experiment == "verb_color":
        # stem: strategy_idx_verb_color_fixedshape  (fixedshape always 'cube')
        # attr = color (e.g. 'red'), shape = fixed_shape ('cube')
        color = attr  # parts[3]
        if color not in COLOR_SET:
            raise ValueError(f"unknown color in stem: {stem}")
        return f"{VERB_TO_EN[verb]} the {color} cube."
    raise ValueError(experiment)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build stem->task map for attribute experiments including all_factor"
    )
    p.add_argument(
        "--experiment",
        required=True,
        choices=(
            "verb_size",
            "verb_spatial",
            "verb_spatial_new",
            "verb_object",
            "verb_color",
            "object_spatial",
            "color_spatial",
            "color_size",
            "spatial_size",
            "spatial_object",
            "size_object",
            "all_factor",
        ),
    )
    p.add_argument("--demos-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stems-file", type=Path, default=None)
    args = p.parse_args()

    if not args.demos_dir.is_dir():
        raise FileNotFoundError(args.demos_dir)

    if args.stems_file is not None:
        stems = [ln.strip() for ln in args.stems_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        by_stem = {p.stem: p for p in args.demos_dir.rglob("*_rgb.h5")}
        rgb_paths = []
        for s in stems:
            if s not in by_stem:
                raise FileNotFoundError(f"stem from stems-file not found: {s}")
            rgb_paths.append(by_stem[s])
    else:
        rgb_paths = sorted(args.demos_dir.rglob("*_rgb.h5"))
    if not rgb_paths:
        raise FileNotFoundError(f"no *_rgb.h5 under {args.demos_dir}")

    out: dict[str, str] = {}
    skipped = 0
    for pth in rgb_paths:
        try:
            out[pth.stem] = _label(args.experiment, pth.stem)
        except ValueError as e:
            skipped += 1
            print(f"[skip] {pth.stem}: {e}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(out)} entries -> {args.output} (skipped={skipped})")


if __name__ == "__main__":
    main()
