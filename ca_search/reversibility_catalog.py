from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from .binary_catalog import BinaryCatalog
from .reversibility import _decode_uint64_states, _validate_grid
from .simple_filters import detect_rigid_source_position


@dataclass(frozen=True)
class CatalogStageResult:
    stage: str
    tested_rules: int
    rejected_rules: int
    surviving_rules: int


@dataclass(frozen=True)
class CatalogScreenResult:
    binary_path: str
    metadata_path: str
    total_rules: int
    trivial_reversible_rules: int
    nontrivial_rules: int
    stages: list[CatalogStageResult]
    surviving_rule_count: int
    surviving_legacy_indices: list[int]
    surviving_stable_indices: list[int]
    surviving_stable_ids: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _enumerate_sector_codes(num_cells: int, population: int) -> np.ndarray:
    if population < 0 or population > num_cells:
        raise ValueError(f"Invalid population {population} for {num_cells} cells")
    codes = np.zeros(sum(1 for _ in combinations(range(num_cells), population)), dtype=np.uint64)
    for row, combo in enumerate(combinations(range(num_cells), population)):
        code = 0
        for bit in combo:
            code |= 1 << bit
        codes[row] = np.uint64(code)
    return codes


def _enumerate_full_codes(num_cells: int) -> np.ndarray:
    if num_cells > 20:
        raise ValueError("exact full-torus enumeration is currently intended for at most 20 cells")
    return np.arange(1 << num_cells, dtype=np.uint64)


def _precompute_indices(width: int, height: int, state_codes: np.ndarray) -> np.ndarray:
    states = _decode_uint64_states(state_codes, width, height)
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
    return indices.reshape(indices.shape[0], width * height)


def _batch_is_injective(rule_batch: np.ndarray, indices_flat: np.ndarray) -> np.ndarray:
    outputs = rule_batch[:, indices_flat]
    weights = (np.uint64(1) << np.arange(indices_flat.shape[1], dtype=np.uint64))[None, None, :]
    output_codes = np.sum(outputs.astype(np.uint64, copy=False) * weights, axis=2, dtype=np.uint64)
    sorted_codes = np.sort(output_codes, axis=1)
    has_duplicate = np.any(sorted_codes[:, 1:] == sorted_codes[:, :-1], axis=1)
    return ~has_duplicate


def _detect_trivial_reversible_mask(lut_bits: np.ndarray) -> np.ndarray:
    trivial = np.zeros(lut_bits.shape[0], dtype=bool)
    for index, row in enumerate(lut_bits):
        source = detect_rigid_source_position(row.tolist())
        trivial[index] = source is not None
    return trivial


def screen_catalog_exact_reversibility(
    catalog: BinaryCatalog,
    *,
    sector_grid: tuple[int, int] = (4, 4),
    particle_populations: tuple[int, ...] = (2, 3),
    hole_populations: tuple[int, ...] = (1, 2),
    torus_grids: tuple[tuple[int, int], ...] = ((3, 3),),
    rule_batch_size: int = 1024,
) -> CatalogScreenResult:
    trivial_mask = _detect_trivial_reversible_mask(catalog.lut_bits)
    survivor_mask = ~trivial_mask
    stage_results: list[CatalogStageResult] = []

    width, height = sector_grid
    n_cells = _validate_grid(width, height)

    stage_specs: list[tuple[str, np.ndarray]] = []
    for population in particle_populations:
        codes = _enumerate_sector_codes(n_cells, population)
        stage_specs.append((f"particle-sector:{population}@{width}x{height}", _precompute_indices(width, height, codes)))
    for holes in hole_populations:
        codes = _enumerate_sector_codes(n_cells, n_cells - holes)
        stage_specs.append((f"hole-sector:{holes}@{width}x{height}", _precompute_indices(width, height, codes)))
    for torus_width, torus_height in torus_grids:
        num_cells = _validate_grid(torus_width, torus_height)
        codes = _enumerate_full_codes(num_cells)
        stage_specs.append((f"torus:{torus_width}x{torus_height}", _precompute_indices(torus_width, torus_height, codes)))

    for stage_name, indices_flat in stage_specs:
        active_indices = np.nonzero(survivor_mask & ~trivial_mask)[0]
        if len(active_indices) == 0:
            stage_results.append(
                CatalogStageResult(
                    stage=stage_name,
                    tested_rules=0,
                    rejected_rules=0,
                    surviving_rules=0,
                )
            )
            continue

        keep = np.zeros(len(active_indices), dtype=bool)
        tested_rules = len(active_indices)
        for start in range(0, len(active_indices), rule_batch_size):
            stop = min(start + rule_batch_size, len(active_indices))
            chunk_indices = active_indices[start:stop]
            keep[start:stop] = _batch_is_injective(catalog.lut_bits[chunk_indices], indices_flat)

        rejected_rules = int(tested_rules - int(keep.sum()))
        next_survivor_mask = np.zeros_like(survivor_mask)
        next_survivor_mask[active_indices[keep]] = True
        next_survivor_mask |= trivial_mask
        survivor_mask = next_survivor_mask
        stage_results.append(
            CatalogStageResult(
                stage=stage_name,
                tested_rules=tested_rules,
                rejected_rules=rejected_rules,
                surviving_rules=int(np.count_nonzero(survivor_mask & ~trivial_mask)),
            )
        )

    final_indices = np.nonzero(survivor_mask & ~trivial_mask)[0]
    return CatalogScreenResult(
        binary_path=str(catalog.binary_path),
        metadata_path=str(catalog.metadata_path),
        total_rules=len(catalog.ids),
        trivial_reversible_rules=int(np.count_nonzero(trivial_mask)),
        nontrivial_rules=int(np.count_nonzero(~trivial_mask)),
        stages=stage_results,
        surviving_rule_count=len(final_indices),
        surviving_legacy_indices=[int(catalog.ids[idx]) for idx in final_indices],
        surviving_stable_indices=[int(catalog.stable_indices[idx]) for idx in final_indices],
        surviving_stable_ids=[catalog.stable_ids[idx] for idx in final_indices],
    )
