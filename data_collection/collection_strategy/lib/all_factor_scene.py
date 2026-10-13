"""Scene / distractor kwargs for ``all_factor`` VerbObjectColor-v1 collection."""

from __future__ import annotations

import random
from typing import Any


def _relative_size_preset(size_label: str) -> tuple[float, list[float], int]:
    """Same numeric presets as ``collect_pairwise_attribute._size_controls`` (no small/large)."""
    if size_label == "smaller":
        return 0.82, [1.08], 1
    if size_label == "larger":
        return 1.18, [0.92], 1
    if size_label == "smallest":
        return 0.78, [1.00, 1.24], 2
    if size_label == "largest":
        return 1.26, [1.00, 0.80], 2
    raise ValueError(size_label)


def pack_verb_object_color_options(
    *,
    size_label: str,
    rng: random.Random,
    distractor_max_cli: int,
    num_distractors_cli: int,
    small_large_distractor_policy: str = "bernoulli",
    small_large_no_distractor_prob: float = 0.70,
    small_large_fixed_num_distractors: int = 0,
) -> dict[str, Any]:
    """Return kwargs for ``run_verb_color_shape_motion_planning`` (VerbObjectColor-v1).

    - ``small`` / ``large``: distractor policy via ``small_large_distractor_policy``:
      ``bernoulli`` — at most one distractor; ``small_large_no_distractor_prob`` fraction
      of episodes have none (default 0.70 ⇒ ~30% with one).
      ``none`` — always target-only (no distractors).
      ``fixed`` — always ``min(small_large_fixed_num_distractors, cap)`` distractors
      (each slot scaled; twin flags still sampled).
      ``random_uniform`` — integer distractor count uniform on ``{0,…,cap}`` each episode; per-distractor scales i.i.d.
    - ``smaller`` / ``larger``: comparative layouts with ``n_d`` sampled from ``{1,2}`` per episode;
      policy above does not apply.
    - ``smallest`` / ``largest``: fixed 3-object comparative layouts (``n_d=2``); policy above does not apply.
    """
    cap = max(0, min(3, int(distractor_max_cli), int(num_distractors_cli)))
    pol = str(small_large_distractor_policy).lower()
    if pol not in ("bernoulli", "none", "fixed", "random_uniform"):
        raise ValueError(f"unknown small_large_distractor_policy: {small_large_distractor_policy!r}")
    p_no = float(small_large_no_distractor_prob)
    if not (0.0 <= p_no <= 1.0):
        raise ValueError("small_large_no_distractor_prob must be in [0, 1]")
    twin_p = 0.16

    def _flags_for(n_d: int) -> tuple[bool, bool, bool]:
        flags = [False, False, False]
        for i in range(n_d):
            if rng.random() < twin_p:
                flags[i] = True
        return (flags[0], flags[1], flags[2])

    def _pad_scales(scales: list[float]) -> list[float]:
        while len(scales) < 3:
            scales.append(1.0)
        return scales[:3]

    if size_label == "small":
        if pol == "none":
            n_d = 0
        elif pol == "fixed":
            n_d = min(max(0, int(small_large_fixed_num_distractors)), cap)
        elif pol == "random_uniform":
            n_d = int(rng.randint(0, cap))
        else:
            n_d = 0 if rng.random() < p_no else min(1, cap)
        t_scale = 0.72
        if n_d == 0:
            return dict(
                target_size_scale=t_scale,
                distractor_size_scales=None,
                num_distractors=0,
                distractor_match_target_geometry=(False, False, False),
            )
        scales = [rng.uniform(0.90, 1.26) for _ in range(n_d)]
        return dict(
            target_size_scale=t_scale,
            distractor_size_scales=_pad_scales(scales),
            num_distractors=n_d,
            distractor_match_target_geometry=_flags_for(n_d),
        )

    if size_label == "large":
        if pol == "none":
            n_d = 0
        elif pol == "fixed":
            n_d = min(max(0, int(small_large_fixed_num_distractors)), cap)
        elif pol == "random_uniform":
            n_d = int(rng.randint(0, cap))
        else:
            n_d = 0 if rng.random() < p_no else min(1, cap)
        t_scale = 1.34
        if n_d == 0:
            return dict(
                target_size_scale=t_scale,
                distractor_size_scales=None,
                num_distractors=0,
                distractor_match_target_geometry=(False, False, False),
            )
        scales = [rng.uniform(0.64, 1.10) for _ in range(n_d)]
        return dict(
            target_size_scale=t_scale,
            distractor_size_scales=_pad_scales(scales),
            num_distractors=n_d,
            distractor_match_target_geometry=_flags_for(n_d),
        )

    if size_label in ("smaller", "larger", "smallest", "largest"):
        t_scale, d_scales, n_d = _relative_size_preset(size_label)
        if size_label in ("smaller", "larger"):
            # Comparative labels can appear with 1 or 2 distractors.
            n_d = int(rng.randint(1, 2))
            base = list(d_scales)
            if n_d == 2:
                if size_label == "smaller":
                    # Keep distractors larger than the target.
                    base.append(float(rng.uniform(1.00, 1.24)))
                else:
                    # Keep distractors smaller than the target.
                    base.append(float(rng.uniform(0.80, 1.00)))
            d_scales = base[:n_d]
        flags_l = [False, False, False]
        for i in range(n_d):
            if rng.random() < 0.12:
                flags_l[i] = True
        return dict(
            target_size_scale=t_scale,
            distractor_size_scales=_pad_scales(list(d_scales)),
            num_distractors=n_d,
            distractor_match_target_geometry=(flags_l[0], flags_l[1], flags_l[2]),
        )

    raise ValueError(size_label)
