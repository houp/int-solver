"""Amplifier LUTs for 2D density classification.

Each amplifier is a standard binary CA rule of the Moore 3x3 neighborhood.
The neighborhood bit layout (matching the repo's convention) is:

        x  y  z
        t  u  w
        a  b  c

with index = x<<8 | y<<7 | z<<6 | t<<5 | u<<4 | w<<3 | a<<2 | b<<1 | c.

We expose a LUT (list of 512 ints) for each amplifier kind. All amplifiers
here are local majorities over a subset of the 3x3 Moore neighborhood.
Tie handling: `ge_threshold` interprets `sum >= threshold` as 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def _make_lut(predicate: Callable[[int], int]) -> list[int]:
    return [int(predicate(i)) for i in range(512)]


def _bits(i: int) -> tuple[int, int, int, int, int, int, int, int, int]:
    x = (i >> 8) & 1
    y = (i >> 7) & 1
    z = (i >> 6) & 1
    t = (i >> 5) & 1
    u = (i >> 4) & 1
    w = (i >> 3) & 1
    a = (i >> 2) & 1
    b = (i >> 1) & 1
    c = i & 1
    return x, y, z, t, u, w, a, b, c


# Full 3x3 Moore majority: 9 cells, threshold 5.
def moore9_majority() -> list[int]:
    return _make_lut(lambda i: 1 if bin(i).count("1") >= 5 else 0)


# Von Neumann (center + 4 orthogonal): 5 cells, threshold 3.
def vn5_majority() -> list[int]:
    def p(i):
        _, y, _, t, u, w, _, b, _ = _bits(i)
        return 1 if (y + t + u + w + b) >= 3 else 0
    return _make_lut(p)


# Diagonal-5 (center + 4 diagonals): 5 cells, threshold 3.
def diag5_majority() -> list[int]:
    def p(i):
        x, _, z, _, u, _, a, _, c = _bits(i)
        return 1 if (x + z + u + a + c) >= 3 else 0
    return _make_lut(p)


# Horizontal row-3 (center + left + right): 3 cells, threshold 2.
# This is exactly 1D majority-of-3 (rule 232) applied along rows.
def row3_majority() -> list[int]:
    def p(i):
        _, _, _, t, u, w, _, _, _ = _bits(i)
        return 1 if (t + u + w) >= 2 else 0
    return _make_lut(p)


# Vertical column-3 (center + top + bottom): 3 cells, threshold 2.
# This is 1D majority-of-3 applied along columns.
def col3_majority() -> list[int]:
    def p(i):
        _, y, _, _, u, _, _, b, _ = _bits(i)
        return 1 if (y + u + b) >= 2 else 0
    return _make_lut(p)


# Main-diagonal-3 (center + NW + SE): threshold 2.
def maindiag3_majority() -> list[int]:
    def p(i):
        x, _, _, _, u, _, _, _, c = _bits(i)
        return 1 if (x + u + c) >= 2 else 0
    return _make_lut(p)


# Anti-diagonal-3 (center + NE + SW): threshold 2.
def antidiag3_majority() -> list[int]:
    def p(i):
        _, _, z, _, u, _, a, _, _ = _bits(i)
        return 1 if (z + u + a) >= 2 else 0
    return _make_lut(p)


# Ortho-4 (4 orthogonal, excluding center): threshold 3.
# Note: at sum == 2 we return 0 (strict majority on 4 cells requires 3).
def ortho4_majority() -> list[int]:
    def p(i):
        _, y, _, t, _, w, _, b, _ = _bits(i)
        return 1 if (y + t + w + b) >= 3 else 0
    return _make_lut(p)


# Corner-4 (4 diagonals, excluding center): threshold 3.
def corner4_majority() -> list[int]:
    def p(i):
        x, _, z, _, _, _, a, _, c = _bits(i)
        return 1 if (x + z + a + c) >= 3 else 0
    return _make_lut(p)


# Moore-8 (ring of 8, no center): threshold 5.
def moore8_majority() -> list[int]:
    def p(i):
        x, y, z, t, _, w, a, b, c = _bits(i)
        return 1 if (x + y + z + t + w + a + b + c) >= 5 else 0
    return _make_lut(p)


# Dictionary of available amplifier kinds, each mapping name -> LUT bits.
# Cross of 5 along orthogonal axes (center + immediate orthogonal neighbors at
# distance 1, same as VN5). Alias for clarity in alternating schedules.

AMPLIFIERS: dict[str, Callable[[], list[int]]] = {
    "moore9": moore9_majority,
    "vn5": vn5_majority,
    "diag5": diag5_majority,
    "row3": row3_majority,
    "col3": col3_majority,
    "maindiag3": maindiag3_majority,
    "antidiag3": antidiag3_majority,
    "ortho4": ortho4_majority,
    "corner4": corner4_majority,
    "moore8": moore8_majority,
}


def build_radius2_majority_stepper():
    """Return a function that applies a radius-2 rule using numpy rolls.

    Does NOT fit into the 512-entry Moore LUT simulator. Handled separately.
    """
    raise NotImplementedError("use apply_radius2_step directly")


RADIUS_KIND_SPECS = {
    "moore25": ("box", 2, 13),
    "vn13":    ("diamond", 2, 7),
    "moore49": ("box", 3, 25),
    "vn25":    ("diamond", 3, 13),
    "moore81": ("box", 4, 41),
}


def apply_radius2_step(states, kind: str):
    """Apply one step of a radius-2+ majority rule on a numpy array."""
    import numpy as np

    if kind not in RADIUS_KIND_SPECS:
        raise KeyError(f"unknown radius-2 kind: {kind}")
    shape, radius, threshold = RADIUS_KIND_SPECS[kind]
    s = np.asarray(states, dtype=np.int16)
    total = np.zeros_like(s)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if shape == "diamond" and abs(dx) + abs(dy) > radius:
                continue
            total += np.roll(np.roll(s, dy, axis=1), dx, axis=2)
    return (total >= threshold).astype(np.uint8)


def apply_radius_step_mlx(states, kind: str):
    """Apply one step of a radius majority rule on an MLX array.

    `states` is an mlx.core array of dtype uint8, shape (batch, H, W).
    Returns a new uint8 mlx array of the same shape.
    """
    import mlx.core as mx

    if kind not in RADIUS_KIND_SPECS:
        raise KeyError(f"unknown radius kind: {kind}")
    shape, radius, threshold = RADIUS_KIND_SPECS[kind]

    s = states.astype(mx.int16)
    total = mx.zeros_like(s)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if shape == "diamond" and abs(dx) + abs(dy) > radius:
                continue
            total = total + mx.roll(mx.roll(s, dy, axis=1), dx, axis=2)
    return (total >= threshold).astype(mx.uint8)


def build_amplifier(name: str) -> list[int]:
    if name not in AMPLIFIERS:
        raise KeyError(f"unknown amplifier '{name}'; available: {list(AMPLIFIERS)}")
    return AMPLIFIERS[name]()


def build_sequence(names: list[str]) -> list[list[int]]:
    """Return a list of LUTs, one per step in the sequence."""
    return [build_amplifier(n) for n in names]


@dataclass(frozen=True)
class AmplifierSchedule:
    """A cyclic schedule of local-majority amplifiers.

    The schedule applies `sequence[0], sequence[1], ..., sequence[-1]` in
    order, then repeats for `total_steps` total applications.
    """
    name: str
    sequence: tuple[str, ...]
    total_steps: int

    def step_names(self) -> list[str]:
        out = []
        for i in range(self.total_steps):
            out.append(self.sequence[i % len(self.sequence)])
        return out
