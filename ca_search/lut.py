from __future__ import annotations

import hashlib
from typing import Iterable


POSITION_ORDER = ("x", "y", "z", "t", "u", "w", "a", "b", "c")
POSITION_WEIGHTS = {
    "x": 1 << 8,
    "y": 1 << 7,
    "z": 1 << 6,
    "t": 1 << 5,
    "u": 1 << 4,
    "w": 1 << 3,
    "a": 1 << 2,
    "b": 1 << 1,
    "c": 1 << 0,
}

VELOCITY_TO_SOURCE_POSITION = {
    "static": "u",
    "north": "b",
    "south": "y",
    "east": "t",
    "west": "w",
    "northeast": "a",
    "northwest": "c",
    "southeast": "x",
    "southwest": "z",
}
SOURCE_POSITION_TO_VELOCITY = {
    position: velocity for velocity, position in VELOCITY_TO_SOURCE_POSITION.items()
}

NIBBLE_BITS = {
    "0": (0, 0, 0, 0),
    "1": (1, 0, 0, 0),
    "2": (0, 1, 0, 0),
    "3": (1, 1, 0, 0),
    "4": (0, 0, 1, 0),
    "5": (1, 0, 1, 0),
    "6": (0, 1, 1, 0),
    "7": (1, 1, 1, 0),
    "8": (0, 0, 0, 1),
    "9": (1, 0, 0, 1),
    "a": (0, 1, 0, 1),
    "b": (1, 1, 0, 1),
    "c": (0, 0, 1, 1),
    "d": (1, 0, 1, 1),
    "e": (0, 1, 1, 1),
    "f": (1, 1, 1, 1),
}


def neighborhood_index(x: int, y: int, z: int, t: int, u: int, w: int, a: int, b: int, c: int) -> int:
    return (
        (int(x) << 8)
        | (int(y) << 7)
        | (int(z) << 6)
        | (int(t) << 5)
        | (int(u) << 4)
        | (int(w) << 3)
        | (int(a) << 2)
        | (int(b) << 1)
        | int(c)
    )


def lut_hex_to_bits(lut_hex: str) -> list[int]:
    if len(lut_hex) != 128:
        raise ValueError(f"Expected 128 hex digits, got {len(lut_hex)}")
    bits: list[int] = []
    for ch in lut_hex.lower():
        nibble = NIBBLE_BITS.get(ch)
        if nibble is None:
            raise ValueError(f"Invalid LUT hex digit: {ch!r}")
        bits.extend(nibble)
    if len(bits) != 512:
        raise AssertionError("Internal LUT decode error")
    return bits


def lut_bits_to_hex(bits: Iterable[int]) -> str:
    seq = [int(v) for v in bits]
    if len(seq) != 512:
        raise ValueError(f"Expected 512 LUT bits, got {len(seq)}")
    out: list[str] = []
    for i in range(0, 512, 4):
        nibble = seq[i] | (seq[i + 1] << 1) | (seq[i + 2] << 2) | (seq[i + 3] << 3)
        out.append(format(nibble, "x"))
    return "".join(out)


def rule_stable_id_from_hex(lut_hex: str) -> str:
    normalized = lut_hex.lower()
    if len(normalized) != 128:
        raise ValueError(f"Expected 128 hex digits, got {len(normalized)}")
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def rule_stable_id(bits: Iterable[int]) -> str:
    return rule_stable_id_from_hex(lut_bits_to_hex(bits))


def rule_stable_short_id(stable_id: str, length: int = 12) -> str:
    if length <= 0:
        raise ValueError("length must be positive")
    return stable_id[:length]


def singleton_index(position: str) -> int:
    return POSITION_WEIGHTS[position]


def single_zero_index(position: str) -> int:
    return 511 - POSITION_WEIGHTS[position]


def rigid_rule_bits(source_position: str) -> list[int]:
    source_bit = POSITION_ORDER.index(source_position)
    weight = 1 << (8 - source_bit)
    return [1 if index & weight else 0 for index in range(512)]


def identity_rule_bits() -> list[int]:
    return rigid_rule_bits("u")


def rigid_shift_rule_bits(velocity: str) -> list[int]:
    source_position = VELOCITY_TO_SOURCE_POSITION[velocity]
    return rigid_rule_bits(source_position)


def embedded_von_neumann_traffic_rule_bits(velocity: str) -> list[int]:
    if velocity not in {"north", "south", "east", "west"}:
        raise ValueError(f"Unsupported traffic velocity: {velocity}")

    bits = [0] * 512
    for index in range(512):
        x = (index >> 8) & 1
        y = (index >> 7) & 1
        z = (index >> 6) & 1
        t = (index >> 5) & 1
        u = (index >> 4) & 1
        w = (index >> 3) & 1
        a = (index >> 2) & 1
        b = (index >> 1) & 1
        c = index & 1
        _ = (x, z, a, c)  # corners are ignored by the embedded 1D traffic rule

        if velocity == "east":
            value = (t and not u) or (u and w)
        elif velocity == "west":
            value = (w and not u) or (u and t)
        elif velocity == "north":
            value = (b and not u) or (u and y)
        else:  # south
            value = (y and not u) or (u and b)

        bits[index] = int(value)
    return bits
