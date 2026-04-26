from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, islice
from typing import Iterable

import numpy as np

from .lut import rule_stable_id
from .simple_filters import summarize_simple_rule


@dataclass(frozen=True)
class CollisionWitness:
    input_a: int
    input_b: int
    output: int


@dataclass(frozen=True)
class SectorInjectivityResult:
    width: int
    height: int
    population: int
    tested_states: int
    is_injective: bool
    witness: CollisionWitness | None


@dataclass(frozen=True)
class TorusBijectivityResult:
    width: int
    height: int
    tested_states: int
    is_bijective: bool
    witness: CollisionWitness | None


@dataclass(frozen=True)
class ExactReversibilityScreen:
    stable_id: str
    simple_tags: tuple[str, ...]
    trivial_reversible: bool
    sector_results: list[SectorInjectivityResult]
    torus_results: list[TorusBijectivityResult]
    rejection_stage: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_grid(width: int, height: int) -> int:
    n_cells = width * height
    if n_cells <= 0:
        raise ValueError("grid dimensions must be positive")
    if n_cells > 64:
        raise ValueError("exact state encoding currently supports at most 64 cells")
    return n_cells


def _decode_uint64_states(codes: np.ndarray, width: int, height: int) -> np.ndarray:
    n_cells = _validate_grid(width, height)
    shifts = np.arange(n_cells, dtype=np.uint64)
    flat = ((codes[:, None] >> shifts[None, :]) & 1).astype(np.uint8, copy=False)
    return flat.reshape(len(codes), height, width)


def _encode_uint64_states(states: np.ndarray) -> np.ndarray:
    batch, height, width = states.shape
    n_cells = _validate_grid(width, height)
    flat = np.ascontiguousarray(states.reshape(batch, n_cells), dtype=np.uint8)
    packed = np.packbits(flat, axis=1, bitorder="little")
    if packed.shape[1] < 8:
        padding = np.zeros((batch, 8 - packed.shape[1]), dtype=np.uint8)
        packed = np.concatenate((packed, padding), axis=1)
    packed = np.ascontiguousarray(packed[:, :8])
    return packed.view(np.uint64).reshape(batch)


def _step_many_states_single_rule(states: np.ndarray, rule_bits: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.uint8)
    rule = np.asarray(rule_bits, dtype=np.uint8)
    x = np.roll(np.roll(states, 1, axis=1), 1, axis=2)
    y = np.roll(states, 1, axis=1)
    z = np.roll(np.roll(states, 1, axis=1), -1, axis=2)
    t = np.roll(states, 1, axis=2)
    u = states
    w = np.roll(states, -1, axis=2)
    a = np.roll(np.roll(states, -1, axis=1), 1, axis=2)
    b = np.roll(states, -1, axis=1)
    c = np.roll(np.roll(states, -1, axis=1), -1, axis=2)
    indices = (
        (x.astype(np.uint16) << 8)
        | (y.astype(np.uint16) << 7)
        | (z.astype(np.uint16) << 6)
        | (t.astype(np.uint16) << 5)
        | (u.astype(np.uint16) << 4)
        | (w.astype(np.uint16) << 3)
        | (a.astype(np.uint16) << 2)
        | (b.astype(np.uint16) << 1)
        | c.astype(np.uint16)
    )
    return rule[indices]


def _iter_weight_sector_codes(num_cells: int, population: int, batch_size: int):
    if population < 0 or population > num_cells:
        raise ValueError(f"Invalid population {population} for {num_cells} cells")
    iterator = combinations(range(num_cells), population)
    while True:
        chunk = list(islice(iterator, batch_size))
        if not chunk:
            return
        codes = np.zeros(len(chunk), dtype=np.uint64)
        for row, combo in enumerate(chunk):
            code = 0
            for bit in combo:
                code |= 1 << bit
            codes[row] = np.uint64(code)
        yield codes


def test_sector_injective(
    rule_bits: Iterable[int],
    *,
    width: int,
    height: int,
    population: int,
    batch_size: int = 8192,
) -> SectorInjectivityResult:
    n_cells = _validate_grid(width, height)
    rule = np.asarray(list(rule_bits), dtype=np.uint8)
    seen: dict[int, int] = {}
    tested_states = 0

    for input_codes in _iter_weight_sector_codes(n_cells, population, batch_size):
        states = _decode_uint64_states(input_codes, width, height)
        outputs = _step_many_states_single_rule(states, rule)
        output_codes = _encode_uint64_states(outputs)
        tested_states += len(input_codes)
        for input_code, output_code in zip(input_codes.tolist(), output_codes.tolist(), strict=True):
            previous = seen.get(output_code)
            if previous is not None and previous != input_code:
                return SectorInjectivityResult(
                    width=width,
                    height=height,
                    population=population,
                    tested_states=tested_states,
                    is_injective=False,
                    witness=CollisionWitness(input_a=previous, input_b=input_code, output=output_code),
                )
            seen[output_code] = input_code

    return SectorInjectivityResult(
        width=width,
        height=height,
        population=population,
        tested_states=tested_states,
        is_injective=True,
        witness=None,
    )


def test_small_torus_bijective(
    rule_bits: Iterable[int],
    *,
    width: int,
    height: int,
    batch_size: int = 16384,
) -> TorusBijectivityResult:
    n_cells = _validate_grid(width, height)
    if n_cells > 20:
        raise ValueError("exact full-torus bijectivity is currently intended for at most 20 cells")

    total_states = 1 << n_cells
    rule = np.asarray(list(rule_bits), dtype=np.uint8)
    previous_input = np.full(total_states, -1, dtype=np.int64)
    tested_states = 0

    for start in range(0, total_states, batch_size):
        stop = min(start + batch_size, total_states)
        input_codes = np.arange(start, stop, dtype=np.uint64)
        states = _decode_uint64_states(input_codes, width, height)
        outputs = _step_many_states_single_rule(states, rule)
        output_codes = _encode_uint64_states(outputs).astype(np.int64, copy=False)
        tested_states += len(input_codes)

        for input_code, output_code in zip(input_codes.tolist(), output_codes.tolist(), strict=True):
            previous = int(previous_input[output_code])
            if previous >= 0 and previous != input_code:
                return TorusBijectivityResult(
                    width=width,
                    height=height,
                    tested_states=tested_states,
                    is_bijective=False,
                    witness=CollisionWitness(input_a=previous, input_b=input_code, output=output_code),
                )
            previous_input[output_code] = input_code

    return TorusBijectivityResult(
        width=width,
        height=height,
        tested_states=tested_states,
        is_bijective=True,
        witness=None,
    )


def run_exact_reversibility_screen(
    rule_bits: Iterable[int],
    *,
    sector_grid: tuple[int, int] = (8, 8),
    particle_populations: tuple[int, ...] = (2, 3, 4),
    hole_populations: tuple[int, ...] = (1, 2, 3, 4),
    torus_grids: tuple[tuple[int, int], ...] = ((3, 3), (4, 4)),
    sector_batch_size: int = 8192,
    torus_batch_size: int = 16384,
) -> ExactReversibilityScreen:
    bits = list(rule_bits)
    summary = summarize_simple_rule(bits, [])
    simple_tags = summary.tags

    if "identity" in simple_tags or any(tag.startswith("rigid_shift:") for tag in simple_tags):
        return ExactReversibilityScreen(
            stable_id=rule_stable_id(bits),
            simple_tags=simple_tags,
            trivial_reversible=True,
            sector_results=[],
            torus_results=[],
            rejection_stage=None,
        )

    width, height = sector_grid
    n_cells = _validate_grid(width, height)
    sector_results: list[SectorInjectivityResult] = []
    torus_results: list[TorusBijectivityResult] = []

    for population in particle_populations:
        result = test_sector_injective(
            bits,
            width=width,
            height=height,
            population=population,
            batch_size=sector_batch_size,
        )
        sector_results.append(result)
        if not result.is_injective:
            return ExactReversibilityScreen(
                stable_id=rule_stable_id(bits),
                simple_tags=simple_tags,
                trivial_reversible=False,
                sector_results=sector_results,
                torus_results=torus_results,
                rejection_stage=f"particle-sector:{population}",
            )

    for holes in hole_populations:
        result = test_sector_injective(
            bits,
            width=width,
            height=height,
            population=n_cells - holes,
            batch_size=sector_batch_size,
        )
        sector_results.append(result)
        if not result.is_injective:
            return ExactReversibilityScreen(
                stable_id=rule_stable_id(bits),
                simple_tags=simple_tags,
                trivial_reversible=False,
                sector_results=sector_results,
                torus_results=torus_results,
                rejection_stage=f"hole-sector:{holes}",
            )

    for torus_width, torus_height in torus_grids:
        result = test_small_torus_bijective(
            bits,
            width=torus_width,
            height=torus_height,
            batch_size=torus_batch_size,
        )
        torus_results.append(result)
        if not result.is_bijective:
            return ExactReversibilityScreen(
                stable_id=rule_stable_id(bits),
                simple_tags=simple_tags,
                trivial_reversible=False,
                sector_results=sector_results,
                torus_results=torus_results,
                rejection_stage=f"torus:{torus_width}x{torus_height}",
            )

    return ExactReversibilityScreen(
        stable_id=rule_stable_id(bits),
        simple_tags=simple_tags,
        trivial_reversible=False,
        sector_results=sector_results,
        torus_results=torus_results,
        rejection_stage=None,
    )
