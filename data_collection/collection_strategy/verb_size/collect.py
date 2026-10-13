#!/usr/bin/env python3
"""Verb × Size collection wrapper."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    rc = subprocess.call(
        [
            sys.executable,
            "-m",
            "collection_strategy.collect_pairwise_attribute",
            "--experiment",
            "verb_size",
            *sys.argv[1:],
        ],
        cwd=_ROOT,
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
