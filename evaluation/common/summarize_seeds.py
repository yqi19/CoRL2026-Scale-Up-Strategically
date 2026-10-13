#!/usr/bin/env python3
"""Average the ``overall_*=S/N (R%)`` lines of per-seed result files.

Usage: python summarize_seeds.py <dir> [glob=*_seed*.txt]
Prints one line per seed and the mean rate per metric across seeds.
"""
import re
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    pattern = sys.argv[2] if len(sys.argv) > 2 else "*_seed*.txt"
    acc: dict[str, list[float]] = {}
    for p in sorted(root.glob(pattern)):
        line = []
        for m in re.finditer(r"^(overall_\w+)=(\d+)/(\d+) \(([\d.]+)%\)", p.read_text(), re.M):
            k, r = m.group(1), float(m.group(4))
            acc.setdefault(k, []).append(r)
            line.append(f"{k}={m.group(2)}/{m.group(3)} ({r:.1f}%)")
        print(f"{p.name}: " + ("  ".join(line) if line else "PARSE_FAILED"))
    for k, v in acc.items():
        print(f"AVG {k} = {sum(v) / len(v):.1f}%  (n={len(v)} seeds)")


if __name__ == "__main__":
    main()
