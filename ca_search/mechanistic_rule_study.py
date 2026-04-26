from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .binary_catalog import load_binary_catalog
from .focused_rule_study import (
    DEFAULT_BINARY,
    DEFAULT_METADATA,
    SHIFT_BY_NAME,
    _detect_tail_period,
    _make_diagonal_stripes,
    _make_horizontal_stripes,
    _make_single_hole,
    _make_single_particle,
    _make_vertical_interface,
)
from .simple_filters import summarize_simple_rule
from .simulator import MLXBackend, NumpyBackend


def _backend(name: str):
    if name == "mlx":
        return MLXBackend()
    return NumpyBackend()


def _roll(state: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(state, dy, axis=0), dx, axis=1)


def _bbox(state: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(state)
    if len(xs) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _bbox_area(state: np.ndarray) -> int:
    box = _bbox(state)
    if box is None:
        return 0
    y0, y1, x0, x1 = box
    return int((y1 - y0 + 1) * (x1 - x0 + 1))


def _cluster_count(state: np.ndarray) -> int:
    state = np.asarray(state, dtype=np.uint8)
    visited = np.zeros_like(state, dtype=bool)
    height, width = state.shape
    count = 0
    for y in range(height):
        for x in range(width):
            if state[y, x] == 0 or visited[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if state[ny, nx] == 0 or visited[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return count


def _coords_to_state(height: int, width: int, coords: list[tuple[int, int]], value: int = 1, fill: int = 0) -> np.ndarray:
    state = np.full((height, width), fill, dtype=np.uint8)
    for y, x in coords:
        state[y % height, x % width] = value
    return state


def _best_shift_reference(previous: np.ndarray, current: np.ndarray) -> tuple[str, float]:
    best_name = "static"
    best_overlap = -1.0
    for name, (dy, dx) in SHIFT_BY_NAME.items():
        overlap = float(np.mean(_roll(previous, dy, dx) == current))
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
    return best_name, best_overlap


def _simulate_sequence(backend_name: str, rule_bits: np.ndarray, initial: np.ndarray, steps: int) -> list[np.ndarray]:
    backend = _backend(backend_name)
    rule = backend.asarray(rule_bits[None, ...], dtype="uint8")
    states = backend.asarray(initial[None, ...], dtype="uint8")
    out = [initial.copy()]
    for _ in range(steps):
        states = backend.step_pairwise(states, rule)
        out.append(np.asarray(states)[0].copy())
    return out


@dataclass(frozen=True)
class PairCollisionSummary:
    offset: tuple[int, int]
    collision: bool
    first_collision_step: int | None
    first_separation_step: int | None
    final_matches_free_transport: bool
    final_overlap_with_free_transport: float
    exact_tail_period: int | None
    final_cluster_count: int
    max_bbox_area: int
    final_bbox_area: int
    final_population: int


@dataclass(frozen=True)
class InterfaceSummary:
    name: str
    steps: int
    dominant_shift: str
    dominant_overlap: float
    exact_tail_period: int | None
    mean_interface_width: float
    max_interface_width: int
    late_mean_interface_width: float
    mean_activity: float
    late_mean_activity: float


@dataclass(frozen=True)
class MotifLongRunSummary:
    name: str
    background: str
    steps: int
    exact_tail_period: int | None
    dominant_shift: str
    dominant_overlap: float
    final_cluster_count: int
    max_bbox_area: int
    final_bbox_area: int
    population_constant: bool


@dataclass(frozen=True)
class RuleMechanisticStudy:
    id: int
    stable_index: int
    stable_id: str
    mask: int
    properties: tuple[str, ...]
    tags: tuple[str, ...]
    isolated_particle_velocity: str | None
    isolated_hole_velocity: str | None
    particle_pair_collisions: list[PairCollisionSummary]
    hole_pair_collisions: list[PairCollisionSummary]
    interfaces: list[InterfaceSummary]
    longrun_motifs: list[MotifLongRunSummary]


def _free_transport_state(height: int, width: int, initial_coords: list[tuple[int, int]], velocity: str, step: int, fill: int = 0, value: int = 1) -> np.ndarray:
    dy, dx = SHIFT_BY_NAME[velocity]
    moved = [(y + step * dy, x + step * dx) for y, x in initial_coords]
    return _coords_to_state(height, width, moved, value=value, fill=fill)


def _pair_collision_study(
    backend_name: str,
    rule_bits: np.ndarray,
    offsets: list[tuple[int, int]],
    isolated_velocity: str,
    *,
    background: str,
) -> list[PairCollisionSummary]:
    if isolated_velocity is None:
        return []

    height = width = 129
    origin = (height // 2, width // 2)
    steps = 96
    if background == "zeros":
        fill = 0
        value = 1
    else:
        fill = 1
        value = 0

    summaries: list[PairCollisionSummary] = []
    for offset in offsets:
        coords = [origin, (origin[0] + offset[0], origin[1] + offset[1])]
        initial = _coords_to_state(height, width, coords, value=value, fill=fill)
        seq = _simulate_sequence(backend_name, rule_bits, initial, steps)
        free_seq = [
            _free_transport_state(height, width, coords, isolated_velocity, t, fill=fill, value=value)
            for t in range(steps + 1)
        ]

        first_collision = None
        first_separation = None
        in_collision = False
        max_bbox_area = 0
        for t, state in enumerate(seq):
            box = _bbox(state if background == "zeros" else 1 - state)
            if box is not None:
                y0, y1, x0, x1 = box
                area = (y1 - y0 + 1) * (x1 - x0 + 1)
                max_bbox_area = max(max_bbox_area, int(area))
            overlap = float(np.mean(state == free_seq[t]))
            if overlap < 1.0 and first_collision is None:
                first_collision = t
                in_collision = True
            elif in_collision and overlap == 1.0 and first_separation is None:
                first_separation = t
                in_collision = False

        final_overlap = float(np.mean(seq[-1] == free_seq[-1]))
        active_final = seq[-1] if background == "zeros" else 1 - seq[-1]
        active_seq = [state if background == "zeros" else 1 - state for state in seq]
        summaries.append(
            PairCollisionSummary(
                offset=offset,
                collision=first_collision is not None,
                first_collision_step=first_collision,
                first_separation_step=first_separation,
                final_matches_free_transport=final_overlap == 1.0,
                final_overlap_with_free_transport=final_overlap,
                exact_tail_period=_detect_tail_period(seq, max_period=24, required_cycles=2),
                final_cluster_count=_cluster_count(active_final),
                max_bbox_area=max_bbox_area,
                final_bbox_area=_bbox_area(active_final),
                final_population=int(active_final.sum()),
            )
        )
    return summaries


def _interface_widths(name: str, state: np.ndarray) -> int:
    if name == "vertical_interface":
        profile = state.mean(axis=0)
    elif name == "horizontal_interface":
        profile = state.mean(axis=1)
    elif name == "diagonal_stripes":
        height, width = state.shape
        sums = [[] for _ in range(height + width - 1)]
        for y in range(height):
            for x in range(width):
                sums[y + x].append(int(state[y, x]))
        profile = np.array([np.mean(items) for items in sums], dtype=float)
    else:
        raise ValueError(name)
    return int(np.sum((profile > 0.05) & (profile < 0.95)))


def _interface_study(backend_name: str, rule_bits: np.ndarray, name: str, initial: np.ndarray, steps: int) -> InterfaceSummary:
    seq = _simulate_sequence(backend_name, rule_bits, initial, steps)
    previous = seq[:-1]
    current = seq[1:]
    shift_names = []
    overlaps = []
    widths = []
    activities = []
    for prev, cur in zip(previous, current):
        shift, overlap = _best_shift_reference(prev, cur)
        shift_names.append(shift)
        overlaps.append(overlap)
        widths.append(_interface_widths(name, cur))
        activities.append(float(np.mean(prev != cur)))
    late = slice(steps // 2, None)
    dominant = max(set(shift_names[late]), key=lambda item: shift_names[late].count(item))
    return InterfaceSummary(
        name=name,
        steps=steps,
        dominant_shift=dominant,
        dominant_overlap=float(np.mean(overlaps[late])),
        exact_tail_period=_detect_tail_period(seq, max_period=32, required_cycles=3),
        mean_interface_width=float(np.mean(widths)),
        max_interface_width=int(max(widths)),
        late_mean_interface_width=float(np.mean(widths[late])),
        mean_activity=float(np.mean(activities)),
        late_mean_activity=float(np.mean(activities[late])),
    )


def _motif_longrun(backend_name: str, rule_bits: np.ndarray, name: str, initial: np.ndarray, steps: int, background: str) -> MotifLongRunSummary:
    seq = _simulate_sequence(backend_name, rule_bits, initial, steps)
    previous = seq[:-1]
    current = seq[1:]
    shifts = []
    overlaps = []
    active_seq = [state if background == "zeros" else 1 - state for state in seq]
    for prev, cur in zip(previous, current):
        shift, overlap = _best_shift_reference(prev, cur)
        shifts.append(shift)
        overlaps.append(overlap)
    dominant = max(set(shifts[steps // 2 :]), key=lambda item: shifts[steps // 2 :].count(item))
    populations = [int(state.sum()) for state in active_seq]
    bbox_areas = [_bbox_area(state) for state in active_seq]
    final_active = active_seq[-1]
    return MotifLongRunSummary(
        name=name,
        background=background,
        steps=steps,
        exact_tail_period=_detect_tail_period(seq, max_period=32, required_cycles=2),
        dominant_shift=dominant,
        dominant_overlap=float(np.mean(overlaps[steps // 2 :])),
        final_cluster_count=_cluster_count(final_active),
        max_bbox_area=int(max(bbox_areas)),
        final_bbox_area=int(bbox_areas[-1]),
        population_constant=all(value == populations[0] for value in populations),
    )


def analyze_rule(backend_name: str, rule_bits: np.ndarray, particle_velocity: str | None, hole_velocity: str | None) -> tuple[list[PairCollisionSummary], list[PairCollisionSummary], list[InterfaceSummary], list[MotifLongRunSummary]]:
    offsets = [
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (-1, 2),
        (-2, 1),
        (3, 0),
        (0, 4),
        (4, 0),
    ]
    particle_pairs = _pair_collision_study(
        backend_name, rule_bits, offsets, particle_velocity, background="zeros"
    )
    hole_pairs = _pair_collision_study(
        backend_name, rule_bits, offsets, hole_velocity, background="ones"
    )
    interfaces = [
        _interface_study(backend_name, rule_bits, "vertical_interface", _make_vertical_interface(256, 256), 384),
        _interface_study(backend_name, rule_bits, "horizontal_interface", _make_horizontal_stripes(256, 256), 384),
        _interface_study(backend_name, rule_bits, "diagonal_stripes", _make_diagonal_stripes(256, 256), 384),
    ]
    motifs = [
        _motif_longrun(backend_name, rule_bits, "single_particle", _make_single_particle(129, 129), 192, "zeros"),
        _motif_longrun(backend_name, rule_bits, "single_hole", _make_single_hole(129, 129), 192, "ones"),
    ]
    return particle_pairs, hole_pairs, interfaces, motifs


def run_study(binary_path: Path, metadata_path: Path, candidate_refs: list[str], backend_name: str) -> dict:
    catalog = load_binary_catalog(binary_path, metadata_path)
    reports = []
    for candidate_ref in candidate_refs:
        idx = catalog.resolve_rule_ref(candidate_ref)
        properties = catalog.property_names_for_mask(int(catalog.masks[idx]))
        summary = summarize_simple_rule(catalog.lut_bits[idx].tolist(), properties)
        particle_pairs, hole_pairs, interfaces, motifs = analyze_rule(
            backend_name,
            catalog.lut_bits[idx],
            summary.isolated_particle_velocity,
            summary.isolated_hole_velocity,
        )
        reports.append(
            RuleMechanisticStudy(
                id=int(catalog.ids[idx]),
                stable_index=int(catalog.stable_indices[idx]),
                stable_id=catalog.stable_ids[idx],
                mask=int(catalog.masks[idx]),
                properties=properties,
                tags=summary.tags,
                isolated_particle_velocity=summary.isolated_particle_velocity,
                isolated_hole_velocity=summary.isolated_hole_velocity,
                particle_pair_collisions=particle_pairs,
                hole_pair_collisions=hole_pairs,
                interfaces=interfaces,
                longrun_motifs=motifs,
            )
        )
    return {
        "binary": str(binary_path),
        "metadata": str(metadata_path),
        "backend": backend_name,
        "candidate_refs": candidate_refs,
        "reports": [asdict(report) for report in reports],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mechanistic study of shortlisted CA rules.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--backend", choices=("numpy", "mlx"), default="numpy")
    parser.add_argument("--ids", type=str, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("mechanistic_rule_study.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    result = run_study(args.binary, args.metadata, args.ids, args.backend)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
