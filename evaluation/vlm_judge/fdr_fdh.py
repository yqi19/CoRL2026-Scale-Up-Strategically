#!/usr/bin/env python3
"""Compute FDR per factor pair and the FDH (Copeland) ranking.

FDR (Factor Dominance Rate) for a pair (f1, f2):

    FDR = (N_f1 - N_f2) / (N_f1 + N_f2)

N_f1 / N_f2 are the number of conflict rollouts in which the policy followed
factor 1 / factor 2. FDR > 0 means f1 dominates; rollouts labelled "neither" or
"error" are ignored.

Sources of N_f1 / N_f2:
  --vlm-dir  DIR   VLM judge output (vlm_eval_<experiment>.jsonl from vlm_eval.py),
                   counted from the ``vlm_factor_followed`` label (paper setting).
  --env-dir  DIR   conflict results files (<experiment>/<experiment>_ood_seed*.txt from
                   common/run_conflict.sh), using the env's dual-axis success
                   ``overall_<f1>_success`` / ``overall_<f2>_success``.

FDH (Factor Dominance Hierarchy): Copeland ranking over the pairwise FDRs.
For each pair, f1 wins if FDR > tau, f2 wins if FDR < -tau, otherwise tie.
Copeland score = wins - losses; factors are ranked by score (tau = 0.05).

Factor names: verb, color, object, size, spatial ("shape" in the logs is the
object factor).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter

PAIRS = {
    "verb_color": ("verb", "color"), "verb_object": ("verb", "shape"),
    "verb_size": ("verb", "size"), "verb_spatial": ("verb", "spatial"),
    "color_object": ("color", "shape"), "size_object": ("size", "shape"),
    "color_size": ("color", "size"), "color_spatial": ("color", "spatial"),
    "spatial_size": ("spatial", "size"), "spatial_object": ("spatial", "shape"),
}
FACTOR_NAME = {"shape": "object"}


def _norm(label: str) -> str:
    return label.strip().strip("[]").strip().strip("\"'").strip().lower()


def counts_from_vlm(vlm_dir: pathlib.Path, exp: str, normalize: bool):
    path = vlm_dir / f"vlm_eval_{exp}.jsonl"
    if not path.exists():
        return None
    c = Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        lab = json.loads(line).get("vlm_factor_followed", "error")
        c[_norm(lab) if normalize else lab] += 1
    f1, f2 = PAIRS[exp]
    return c.get(f1, 0), c.get(f2, 0), sum(c.values())


def counts_from_env(env_dir: pathlib.Path, exp: str):
    f1, f2 = PAIRS[exp]
    n1 = n2 = 0
    files = sorted((env_dir / exp).glob(f"{exp}_ood_seed*.txt"))
    if not files:
        return None
    for p in files:
        txt = p.read_text()
        m1 = re.search(rf"^overall_{f1}_success=(\d+)/(\d+)", txt, re.M)
        m2 = re.search(rf"^overall_{f2}_success=(\d+)/(\d+)", txt, re.M)
        if m1 and m2:
            n1 += int(m1.group(1))
            n2 += int(m2.group(1))
    return n1, n2, None


def fdr(n1: int, n2: int) -> float | None:
    return (n1 - n2) / (n1 + n2) if (n1 + n2) > 0 else None


def copeland(fdrs: dict[str, float], tau: float):
    score: dict[str, int] = {}
    for exp, v in fdrs.items():
        a, b = (FACTOR_NAME.get(x, x) for x in PAIRS[exp])
        score.setdefault(a, 0)
        score.setdefault(b, 0)
        if v > tau:
            score[a] += 1
            score[b] -= 1
        elif v < -tau:
            score[b] += 1
            score[a] -= 1
    return sorted(score.items(), key=lambda kv: -kv[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--vlm-dir", type=pathlib.Path)
    src.add_argument("--env-dir", type=pathlib.Path)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--normalize-labels", action="store_true",
                    help="strip brackets/quotes from VLM labels (e.g. '[\"color\"]' -> 'color')")
    a = ap.parse_args()

    fdrs = {}
    print(f"{'experiment':<16}{'f1/f2':<16}{'N_f1':>6}{'N_f2':>6}{'FDR':>9}")
    for exp, (f1, f2) in PAIRS.items():
        if a.vlm_dir:
            r = counts_from_vlm(a.vlm_dir, exp, a.normalize_labels)
        else:
            r = counts_from_env(a.env_dir, exp)
        if r is None:
            continue
        n1, n2, _ = r
        v = fdr(n1, n2)
        name = f"{FACTOR_NAME.get(f1, f1)}/{FACTOR_NAME.get(f2, f2)}"
        print(f"{exp:<16}{name:<16}{n1:>6}{n2:>6}{(f'{v:+.3f}' if v is not None else 'NA'):>9}")
        if v is not None:
            fdrs[exp] = v

    print(f"\nFDH (Copeland, tau={a.tau:.2f}; {len(fdrs)} pairs):")
    for rank, (factor, s) in enumerate(copeland(fdrs, a.tau), 1):
        print(f"  {rank}. {factor:<8} score={s:+d}")


if __name__ == "__main__":
    main()
