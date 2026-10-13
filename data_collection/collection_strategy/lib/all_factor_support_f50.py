"""Support sets for ``all_factor`` collection (verb×color×shape×spatial×size).

Search space: 6×6×6×5×6 = 6480 discrete tuples (see ``CAPS`` / ``tuple_to_flat``).

- ``random``: uniform random cells from the full space (f12 → 12 cells; f20 / f40 use subspaces).
- ``Lrandom``: five single-factor rays through the origin + uniform random extras.
- ``stairvss``: staircase walk + verb/size/shape rays anchored at color=5 (black), spatial=0 (left)
  + 1 random cell.
"""

from __future__ import annotations

import random

# (verb, color, shape, spatial, size) index caps — matches training vocab sizes.
# verb:    lift=0 grasp=1 push=2 pull=3 rotate=4 slide=5
# color:   red=0  yellow=1 blue=2 orange=3 green=4 black=5
# shape:   cube=0 sphere=1 cup=2 car=3 pyramid=4 star=5
# spatial: left=0 right=1 middle=2 front=3 behind=4
# size:    small=0 large=1 smaller=2 larger=3 smallest=4 largest=5
CAPS = (6, 6, 6, 5, 6)
N_SPACE = 6 * 6 * 6 * 5 * 6

CAPS_F20: tuple[int, int, int, int, int] = (3, 3, 3, 3, 2)
N_SPACE_F20 = CAPS_F20[0] * CAPS_F20[1] * CAPS_F20[2] * CAPS_F20[3] * CAPS_F20[4]
CAPS_F40: tuple[int, int, int, int, int] = (5, 5, 5, 5, 4)
N_SPACE_F40 = CAPS_F40[0] * CAPS_F40[1] * CAPS_F40[2] * CAPS_F40[3] * CAPS_F40[4]


def tuple_to_flat(p: tuple[int, int, int, int, int]) -> int:
    v, c, sh, sp, sz = p
    return (((v * 6 + c) * 6 + sh) * 5 + sp) * 6 + sz


def flat_to_tuple(x: int) -> tuple[int, int, int, int, int]:
    if not (0 <= x < N_SPACE):
        raise ValueError(x)
    sz = x % 6
    x //= 6
    sp = x % 5
    x //= 5
    sh = x % 6
    x //= 6
    c = x % 6
    v = x // 6
    return (v, c, sh, sp, sz)


def _dedupe_ordered(rows: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int, int, int]]:
    seen: set[int] = set()
    out: list[tuple[int, int, int, int, int]] = []
    for t in rows:
        k = tuple_to_flat(t)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _tuple_to_flat_caps(p: tuple[int, int, int, int, int], caps: tuple[int, int, int, int, int]) -> int:
    v, c, sh, sp, sz = p
    c0, c1, c2, c3, c4 = caps
    return (((v * c1 + c) * c2 + sh) * c3 + sp) * c4 + sz


def _flat_to_tuple_caps(x: int, caps: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
    c0, c1, c2, c3, c4 = caps
    sz = x % c4; x //= c4
    sp = x % c3; x //= c3
    sh = x % c2; x //= c2
    c  = x % c1; x //= c1
    v  = x % c0
    return (v, c, sh, sp, sz)


def _dedupe_ordered_caps(
    rows: list[tuple[int, int, int, int, int]],
    caps: tuple[int, int, int, int, int],
) -> list[tuple[int, int, int, int, int]]:
    seen: set[int] = set()
    out: list[tuple[int, int, int, int, int]] = []
    for t in rows:
        k = _tuple_to_flat_caps(t, caps)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def random_f20_subspace_indices(rng: random.Random) -> list[tuple[int, int, int, int, int]]:
    """f20: 5 uniform samples without replacement in ``3×3×3×3×2``."""
    chosen = rng.sample(range(N_SPACE_F20), 5)
    return [_flat_to_tuple_caps(k, CAPS_F20) for k in chosen]


def random_f40_subspace_indices(rng: random.Random) -> list[tuple[int, int, int, int, int]]:
    """f40: 10 uniform samples without replacement in ``5×5×5×5×4``."""
    chosen = rng.sample(range(N_SPACE_F40), 10)
    return [_flat_to_tuple_caps(k, CAPS_F40) for k in chosen]


def random_indices(rng: random.Random, n_pick: int) -> list[tuple[int, int, int, int, int]]:
    """random f12: ``n_pick`` uniformly random points in full space (pure random, no greedy dispersion)."""
    n_pick = max(1, min(N_SPACE, int(n_pick)))
    chosen = rng.sample(range(N_SPACE), n_pick)
    return [flat_to_tuple(k) for k in chosen]


def _rays_through_base(
    B: tuple[int, int, int, int, int],
) -> list[tuple[int, int, int, int, int]]:
    v0, c0, sh0, sp0, sz0 = B
    out: list[tuple[int, int, int, int, int]] = []
    for v in range(CAPS[0]):
        out.append((v, c0, sh0, sp0, sz0))
    for c in range(CAPS[1]):
        out.append((v0, c, sh0, sp0, sz0))
    for sh in range(CAPS[2]):
        out.append((v0, c0, sh, sp0, sz0))
    for sp in range(CAPS[3]):
        out.append((v0, c0, sh0, sp, sz0))
    for sz in range(CAPS[4]):
        out.append((v0, c0, sh0, sp0, sz))
    return out


def lrandom_f50_indices(rng: random.Random) -> list[tuple[int, int, int, int, int]]:
    """f50 / Lrandom: five single-factor rays through origin + 4 uniform random points."""
    B = (0, 0, 0, 0, 0)
    rows = _rays_through_base(B)
    have = {tuple_to_flat(t) for t in rows}
    pool = [k for k in range(N_SPACE) if k not in have]
    k_extra = min(4, len(pool))
    rows.extend(flat_to_tuple(k) for k in rng.sample(pool, k_extra))
    return _dedupe_ordered(rows)


def _spiral_walk(B0: tuple[int, int, int, int, int]) -> list[tuple[int, int, int, int, int]]:
    """Monotone staircase walk from ``B0`` until all axes hit their upper bounds.

    Each step increments exactly one axis by +1; axis priority rotates
    size → spatial → shape → color → verb.
    """
    cur = list(B0)
    out: list[tuple[int, int, int, int, int]] = [tuple(cur)]
    axis_order = (4, 3, 2, 1, 0)
    ptr = 0
    upper = [c - 1 for c in CAPS]
    while tuple(cur) != tuple(upper):
        tried = 0
        advanced = False
        while tried < len(axis_order):
            ax = axis_order[ptr % len(axis_order)]
            ptr += 1
            tried += 1
            if cur[ax] < upper[ax]:
                cur[ax] += 1
                out.append(tuple(cur))
                advanced = True
                break
        if not advanced:
            break
    return out


def _rays_verb_size_shape(
    caps: tuple[int, int, int, int, int],
    color: int = 5,
    spatial: int = 0,
) -> list[tuple[int, int, int, int, int]]:
    """Three single-factor rays (verb / size / shape vary); color and spatial fixed at the anchor.

    Default anchor color=5 (black), spatial=0 (left), used by ``stairvss``.
    Each ray starts at (verb=0, color, shape=0, spatial, size=0).
    """
    c0, sp0 = color, spatial
    rows: list[tuple[int, int, int, int, int]] = []
    # Ray 1: verb varies, others anchored at (verb=*, c0, shape=0, sp0, size=0)
    for v in range(caps[0]):
        rows.append((v, c0, 0, sp0, 0))
    # Ray 2: size varies, others anchored at (verb=0, c0, shape=0, sp0, size=*)
    for sz in range(caps[4]):
        rows.append((0, c0, 0, sp0, sz))
    # Ray 3: shape varies, others anchored at (verb=0, c0, shape=*, sp0, size=0)
    for sh in range(caps[2]):
        rows.append((0, c0, sh, sp0, 0))
    return rows


def stairvss_f20_indices(rng: random.Random) -> list[tuple[int, int, int, int, int]]:
    """f20 / stairvss: stair walk + verb/size/shape rays in f20 subspace."""
    _ = rng
    caps = CAPS_F20
    rows = _dedupe_ordered_caps(_rays_verb_size_shape(caps), caps)
    cur = [0, 0, 0, 0, 0]
    while True:
        changed = False
        for a in (4, 3, 2, 1, 0):
            if cur[a] < caps[a] - 1:
                cur[a] += 1
                rows.append(tuple(cur))
                changed = True
        if not changed:
            break
    return _dedupe_ordered_caps(rows, caps)


def stairvss_f40_indices(rng: random.Random) -> list[tuple[int, int, int, int, int]]:
    """f40 / stairvss: stair walk + verb/size/shape rays in f40 subspace."""
    _ = rng
    caps = CAPS_F40
    rows = _dedupe_ordered_caps(_rays_verb_size_shape(caps), caps)
    cur = [0, 0, 0, 0, 0]
    while True:
        changed = False
        for a in (4, 3, 2, 1, 0):
            if cur[a] < caps[a] - 1:
                cur[a] += 1
                rows.append(tuple(cur))
                changed = True
        if not changed:
            break
    return _dedupe_ordered_caps(rows, caps)


def stairvss_f50_indices(rng: random.Random) -> list[tuple[int, int, int, int, int]]:
    """f50 / stairvss: spiral_walk + verb/size/shape rays (anchor color=5/black, spatial=0/left) + 1 random cell."""
    B0 = (0, 0, 0, 0, 0)
    walk = _spiral_walk(B0)
    rays = _rays_verb_size_shape(CAPS, color=5, spatial=0)
    seq = _dedupe_ordered(list(walk) + rays)
    have = {tuple_to_flat(t) for t in seq}
    pool = [k for k in range(N_SPACE) if k not in have]
    if pool:
        seq = _dedupe_ordered(seq + [flat_to_tuple(rng.choice(pool))])
    return seq


ALL_FACTOR_SIZE_LABELS: tuple[str, ...] = (
    "small",
    "large",
    "smaller",
    "larger",
    "smallest",
    "largest",
)


def parse_size_pool_labels(s: str) -> frozenset[int]:
    labels = [x.strip() for x in str(s).split(",") if x.strip()]
    if not labels:
        raise ValueError("empty size pool")
    idxs: list[int] = []
    for lab in labels:
        if lab not in ALL_FACTOR_SIZE_LABELS:
            raise ValueError(f"unknown size label {lab!r} in size pool")
        idxs.append(ALL_FACTOR_SIZE_LABELS.index(lab))
    return frozenset(idxs)


def remap_support_size_pool(
    support: list[tuple[int, int, int, int, int]],
    allowed_sz: frozenset[int],
    rng: random.Random,
) -> list[tuple[int, int, int, int, int]]:
    allowed_list = sorted(allowed_sz)
    if not allowed_list:
        raise ValueError("size pool empty")

    def clamp_sz(sz: int) -> int:
        if sz in allowed_sz:
            return sz
        return int(rng.choice(allowed_list))

    target_n = len(support)
    rows = [(v, c, sh, sp, clamp_sz(sz)) for v, c, sh, sp, sz in support]
    seq = _dedupe_ordered(rows)
    have = {tuple_to_flat(t) for t in seq}

    def random_fill_tuple() -> tuple[int, int, int, int, int]:
        return (
            rng.randrange(6),
            rng.randrange(6),
            rng.randrange(6),
            rng.randrange(5),
            int(rng.choice(allowed_list)),
        )

    attempts = 0
    while len(seq) < target_n and attempts < 100_000:
        t = random_fill_tuple()
        fk = tuple_to_flat(t)
        if fk not in have:
            seq.append(t)
            have.add(fk)
        attempts += 1
    if len(seq) < target_n:
        raise RuntimeError(
            f"remap_support_size_pool: could not reach {target_n} unique tuples (got {len(seq)}); "
            f"pool={allowed_list}"
        )
    return seq[:target_n]


def lrandom_f12_indices(rng: random.Random) -> list[tuple[int, int, int, int, int]]:
    """Lrandom @ f12: verb-axis ray (6 pts) + color-axis ray minus origin (5 pts) + 1 random interior.

    Total: exactly 12 unique points covering two axes plus one exploratory cell.
    """
    out: list[tuple[int, int, int, int, int]] = []
    # verb axis through origin: (v, 0, 0, 0, 0) for v in 0..5
    for v in range(CAPS[0]):
        out.append((v, 0, 0, 0, 0))
    # color axis (skip origin): (0, c, 0, 0, 0) for c in 1..5
    for c in range(1, CAPS[1]):
        out.append((0, c, 0, 0, 0))
    # 1 random from interior (not on either ray)
    have = {tuple_to_flat(t) for t in out}
    pool = [k for k in range(N_SPACE) if k not in have]
    if pool:
        out.append(flat_to_tuple(rng.choice(pool)))
    return out  # 6 + 5 + 1 = 12 pts


def build_support(
    strategy: str,
    rng: random.Random,
    factor_count: int | None = None,
) -> list[tuple[int, int, int, int, int]]:
    if strategy == "random":
        fc = 12 if factor_count is None else int(factor_count)
        if fc == 20:
            return random_f20_subspace_indices(rng)
        if fc == 40:
            return random_f40_subspace_indices(rng)
        return random_indices(rng, fc)
    if strategy == "Lrandom":
        fc = 50 if factor_count is None else int(factor_count)
        if fc == 12:
            return lrandom_f12_indices(rng)
        return lrandom_f50_indices(rng)
    if strategy == "stairvss":
        fci = int(factor_count) if factor_count is not None else 50
        if fci == 20:
            return stairvss_f20_indices(rng)
        if fci == 40:
            return stairvss_f40_indices(rng)
        return stairvss_f50_indices(rng)
    raise ValueError(strategy)
