from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .lut import lut_bits_to_hex, rule_stable_id_from_hex
from .reversibility_catalog import _enumerate_sector_codes, _precompute_indices
from .simple_filters import summarize_simple_rule


RAW_RECORD_BYTES = 64
RAW_RULE_BITS = 510
RAW_STAGE_GRID = (4, 4)
DEFAULT_STAGE_POPULATIONS = (2, 3, 4)


@dataclass(frozen=True)
class RawStageSummary:
    stage: str
    tested_rules: int
    rejected_rules: int
    surviving_rules: int
    unique_rule_bits_touched: int


@dataclass(frozen=True)
class RawSurvivorSummary:
    stable_id: str
    stable_id_short: str
    simple_tags: tuple[str, ...]


@dataclass(frozen=True)
class RawScreenSummary:
    input_paths: list[str]
    total_rules: int
    stages: list[RawStageSummary]
    surviving_rule_count: int
    output_path: str | None
    survivor_summaries_truncated: bool
    survivors: list[RawSurvivorSummary]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _StageSpec:
    name: str
    byte_indices: np.ndarray
    bit_offsets: np.ndarray
    gather_flat: np.ndarray
    const_one_flat: np.ndarray
    rank_lookup: np.ndarray
    num_states: int
    unique_rule_bits_touched: int


def _build_particle_stage_spec(width: int, height: int, population: int) -> _StageSpec:
    state_codes = _enumerate_sector_codes(width * height, population)
    indices = _precompute_indices(width, height, state_codes)

    unique_indices = sorted(set(indices.reshape(-1).tolist()) - {0, 511})
    position = {value: idx for idx, value in enumerate(unique_indices)}

    gather = np.full(indices.shape, -1, dtype=np.int16)
    const_one = indices == 511
    for value, mapped in position.items():
        gather[indices == value] = mapped

    raw_bit_positions = np.asarray([value - 1 for value in unique_indices], dtype=np.int16)
    byte_indices = (raw_bit_positions >> 3).astype(np.int16, copy=False)
    bit_offsets = (raw_bit_positions & 7).astype(np.uint8, copy=False)

    rank_lookup = np.full(1 << 16, -1, dtype=np.int32)
    rank_lookup[state_codes.astype(np.int64, copy=False)] = np.arange(len(state_codes), dtype=np.int32)

    return _StageSpec(
        name=f"particle-sector:{population}@{width}x{height}",
        byte_indices=byte_indices,
        bit_offsets=bit_offsets,
        gather_flat=gather.reshape(-1),
        const_one_flat=const_one.reshape(-1),
        rank_lookup=rank_lookup,
        num_states=len(state_codes),
        unique_rule_bits_touched=len(unique_indices),
    )


def _iter_raw_rule_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.bin"))


def _iter_raw_chunks(path: Path, records_per_chunk: int):
    with path.open("rb") as handle:
        while True:
            buf = np.fromfile(handle, dtype=np.uint8, count=records_per_chunk * RAW_RECORD_BYTES)
            if buf.size == 0:
                return
            if buf.size % RAW_RECORD_BYTES != 0:
                raise ValueError(f"{path} has a truncated raw record payload")
            yield buf.reshape(-1, RAW_RECORD_BYTES)


def _evaluate_stage(records: np.ndarray, spec: _StageSpec) -> np.ndarray:
    if len(records) == 0:
        return np.zeros(0, dtype=bool)

    if len(spec.byte_indices):
        extracted = ((records[:, spec.byte_indices] >> spec.bit_offsets) & 1).astype(np.uint8, copy=False)
    else:
        extracted = np.zeros((len(records), 0), dtype=np.uint8)

    gathered = np.zeros((len(records), spec.gather_flat.size), dtype=np.uint8)
    variable_mask = spec.gather_flat >= 0
    if np.any(variable_mask):
        gathered[:, variable_mask] = extracted[:, spec.gather_flat[variable_mask]]
    if np.any(spec.const_one_flat):
        gathered[:, spec.const_one_flat] = 1

    outputs = gathered.reshape(len(records), spec.num_states, 16)
    weights = (np.uint16(1) << np.arange(16, dtype=np.uint16))[None, None, :]
    output_codes = np.sum(outputs.astype(np.uint16, copy=False) * weights, axis=2, dtype=np.uint16)

    ranks = spec.rank_lookup[output_codes.astype(np.int32, copy=False)]
    valid = np.all(ranks >= 0, axis=1)
    keep = np.zeros(len(records), dtype=bool)
    if np.any(valid):
        valid_indices = np.nonzero(valid)[0]
        sorted_ranks = np.sort(ranks[valid_indices], axis=1)
        unique = np.all(sorted_ranks[:, 1:] != sorted_ranks[:, :-1], axis=1)
        keep[valid_indices] = unique
    return keep


def _decode_raw_record_to_lut_bits(record: np.ndarray) -> list[int]:
    unpacked = np.unpackbits(record, bitorder="little")[:RAW_RULE_BITS].astype(np.uint8, copy=False)
    bits = np.empty(512, dtype=np.uint8)
    bits[0] = 0
    bits[1:511] = unpacked
    bits[511] = 1
    return bits.tolist()


def _summarize_survivor_records(records: list[np.ndarray]) -> list[RawSurvivorSummary]:
    summaries: list[RawSurvivorSummary] = []
    for record in records:
        bits = _decode_raw_record_to_lut_bits(record)
        lut_hex = lut_bits_to_hex(bits)
        stable_id = rule_stable_id_from_hex(lut_hex)
        simple = summarize_simple_rule(bits, [])
        summaries.append(
            RawSurvivorSummary(
                stable_id=stable_id,
                stable_id_short=stable_id[:12],
                simple_tags=simple.tags,
            )
        )
    summaries.sort(key=lambda item: item.stable_id)
    return summaries


def screen_raw_catalog_for_reversibility(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    records_per_chunk: int = 16384,
    stage_populations: Iterable[int] = DEFAULT_STAGE_POPULATIONS,
    max_survivor_summaries: int = 4096,
) -> RawScreenSummary:
    stage_specs = [
        _build_particle_stage_spec(RAW_STAGE_GRID[0], RAW_STAGE_GRID[1], population)
        for population in stage_populations
    ]
    stage_tested = [0 for _ in stage_specs]
    stage_rejected = [0 for _ in stage_specs]
    total_rules = 0
    survivor_records: list[np.ndarray] = []
    surviving_rule_count = 0
    survivor_summaries_truncated = False

    output_handle = None
    output_path_obj = Path(output_path) if output_path is not None else None
    if output_path_obj is not None:
        output_handle = output_path_obj.open("wb")

    try:
        input_paths = _iter_raw_rule_files(input_path)
        for file_path in input_paths:
            for records in _iter_raw_chunks(file_path, records_per_chunk):
                total_rules += len(records)
                current = records
                for index, stage in enumerate(stage_specs):
                    stage_tested[index] += len(current)
                    if len(current) == 0:
                        continue
                    keep = _evaluate_stage(current, stage)
                    stage_rejected[index] += int(len(current) - int(np.count_nonzero(keep)))
                    current = current[keep]
                if len(current) == 0:
                    continue
                surviving_rule_count += len(current)
                if output_handle is not None:
                    output_handle.write(current.tobytes())
                remaining_capacity = max_survivor_summaries - len(survivor_records)
                if remaining_capacity > 0:
                    survivor_records.extend(np.array(record, copy=True) for record in current[:remaining_capacity])
                if len(current) > remaining_capacity:
                    survivor_summaries_truncated = True
    finally:
        if output_handle is not None:
            output_handle.close()

    stage_summaries: list[RawStageSummary] = []
    running_survivors = total_rules
    for tested, rejected, spec in zip(stage_tested, stage_rejected, stage_specs, strict=True):
        running_survivors -= rejected
        stage_summaries.append(
            RawStageSummary(
                stage=spec.name,
                tested_rules=int(tested),
                rejected_rules=int(rejected),
                surviving_rules=int(running_survivors),
                unique_rule_bits_touched=spec.unique_rule_bits_touched,
            )
        )

    return RawScreenSummary(
        input_paths=[str(path) for path in _iter_raw_rule_files(input_path)],
        total_rules=total_rules,
        stages=stage_summaries,
        surviving_rule_count=surviving_rule_count,
        output_path=str(output_path_obj) if output_path_obj is not None else None,
        survivor_summaries_truncated=survivor_summaries_truncated,
        survivors=_summarize_survivor_records(survivor_records),
    )


def write_raw_screen_summary(path: str | Path, summary: RawScreenSummary) -> None:
    Path(path).write_text(json.dumps(summary.to_dict(), indent=2))
