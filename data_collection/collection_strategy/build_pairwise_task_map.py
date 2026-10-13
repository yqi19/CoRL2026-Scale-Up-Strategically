#!/usr/bin/env python3
"""
Build a JSON task map (H5 stem -> language label) for pairwise VerbObjectColor demos.

Scans ``--demos-dir`` for replay outputs ``**/*_rgb.h5``, parses each stem with
``pairwise_task_language`` so every episode gets an instruction matching its
(verb, color, shape) regardless of which pairwise experiment produced it.

Usage (repo root)::

  python collection_strategy/build_pairwise_task_map.py \\
    --demos-dir /path/to/demo_gen/verb_color/random \\
    --output /tmp/task_map.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collection_strategy.lib.pairwise_task_language import traj_stem_to_task_label


def main() -> None:
    p = argparse.ArgumentParser(description="Build stem->task JSON for pairwise RGB H5 files")
    p.add_argument("--demos-dir", type=Path, required=True, help="Root passed to collect_pairwise --record-base/... ")
    p.add_argument("--output", type=Path, required=True, help="Write JSON dict {stem: label}")
    p.add_argument(
        "--stems-file",
        type=Path,
        default=None,
        help="Optional: one H5 stem per line (e.g. ours_0000_lift_red_cup_rgb); only those *_rgb.h5 are included",
    )
    p.add_argument(
        "--skip-unparseable",
        action="store_true",
        help="Skip files whose stems do not match pairwise naming (default: error)",
    )
    args = p.parse_args()
    demos_dir = args.demos_dir
    if not demos_dir.is_dir():
        raise FileNotFoundError(demos_dir)

    task_map: dict[str, str] = {}
    if args.stems_file is not None:
        text = args.stems_file.read_text(encoding="utf-8")
        ordered_stems: list[str] = []
        seen: set[str] = set()
        for ln in text.splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            if s not in seen:
                seen.add(s)
                ordered_stems.append(s)
        if not ordered_stems:
            raise SystemExit("--stems-file is empty")
        by_stem: dict[str, Path] = {}
        for pth in demos_dir.rglob("*_rgb.h5"):
            by_stem[pth.stem] = pth
        rgb_paths = []
        for stem in ordered_stems:
            if stem not in by_stem:
                raise SystemExit(f"--stems-file: no *_rgb.h5 for stem {stem!r}")
            rgb_paths.append(by_stem[stem])
    else:
        rgb_paths = sorted(demos_dir.rglob("*_rgb.h5"))
    if not rgb_paths:
        raise FileNotFoundError(f"no *_rgb.h5 under {demos_dir}")

    errors: list[str] = []
    for h5 in rgb_paths:
        stem = h5.stem
        try:
            task_map[stem] = traj_stem_to_task_label(stem)
        except ValueError as e:
            if args.skip_unparseable:
                continue
            errors.append(f"{h5}: {e}")

    if errors:
        raise SystemExit("Failed to parse stems:\n" + "\n".join(errors))

    if not task_map:
        raise SystemExit("No parseable *_rgb.h5 files")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(task_map, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(task_map)} entries -> {args.output}")


if __name__ == "__main__":
    main()
