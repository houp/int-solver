from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .binary_catalog import load_binary_catalog
from .focused_rule_study import DEFAULT_BINARY, DEFAULT_METADATA, SHIFT_BY_NAME, _detect_tail_period
from .simple_filters import summarize_simple_rule
from .simulator import MLXBackend, NumpyBackend


def _backend(name: str):
    if name == "mlx":
        return MLXBackend()
    return NumpyBackend()


def _roll(state: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(state, dy, axis=0), dx, axis=1)


def _best_shift_reference(previous: np.ndarray, current: np.ndarray) -> tuple[str, float]:
    best_name = "static"
    best_overlap = -1.0
    for name, (dy, dx) in SHIFT_BY_NAME.items():
        overlap = float(np.mean(_roll(previous, dy, dx) == current))
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
    return best_name, best_overlap


def _cluster_count8(state: np.ndarray) -> int:
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
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny = cy + dy
                        nx = cx + dx
                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            continue
                        if state[ny, nx] == 0 or visited[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return count


def _bbox_area(state: np.ndarray) -> int:
    ys, xs = np.nonzero(state)
    if len(xs) == 0:
        return 0
    return int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_y = min(y for y, _ in cells)
    min_x = min(x for _, x in cells)
    normalized = tuple(sorted((y - min_y, x - min_x) for y, x in cells))
    return normalized


def _shape_span(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    ys = [y for y, _ in shape]
    xs = [x for _, x in shape]
    return max(ys) + 1, max(xs) + 1


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    y, x = cell
    return [
        (y + dy, x + dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dy == 0 and dx == 0)
    ]


def enumerate_connected_shapes(max_size: int, max_span: int = 4) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    by_size: dict[int, set[tuple[tuple[int, int], ...]]] = defaultdict(set)
    by_size[1].add(((0, 0),))
    for size in range(2, max_size + 1):
        previous = by_size[size - 1]
        for shape in previous:
            cells = set(shape)
            candidates = set()
            for cell in shape:
                for neighbor in _neighbors8(cell):
                    if neighbor not in cells:
                        candidates.add(neighbor)
            for candidate in candidates:
                expanded = set(cells)
                expanded.add(candidate)
                normalized = _normalize_shape(expanded)
                height, width = _shape_span(normalized)
                if height <= max_span and width <= max_span:
                    by_size[size].add(normalized)
    return {size: sorted(shapes) for size, shapes in by_size.items()}


def _coords_to_state(height: int, width: int, coords: tuple[tuple[int, int], ...], *, fill: int, value: int) -> np.ndarray:
    state = np.full((height, width), fill, dtype=np.uint8)
    y0 = height // 2
    x0 = width // 2
    for y, x in coords:
        state[(y0 + y) % height, (x0 + x) % width] = value
    return state


def _simulate_sequence(backend_name: str, rule_bits: np.ndarray, initial: np.ndarray, steps: int) -> list[np.ndarray]:
    backend = _backend(backend_name)
    rule = backend.asarray(rule_bits[None, ...], dtype="uint8")
    states = backend.asarray(initial[None, ...], dtype="uint8")
    seq = [initial.copy()]
    for _ in range(steps):
        states = backend.step_pairwise(states, rule)
        seq.append(np.asarray(states)[0].copy())
    return seq


def _free_transport_state(height: int, width: int, coords: tuple[tuple[int, int], ...], velocity: str, step: int, *, fill: int, value: int) -> np.ndarray:
    dy, dx = SHIFT_BY_NAME[velocity]
    moved = tuple((y + step * dy, x + step * dx) for y, x in coords)
    return _coords_to_state(height, width, moved, fill=fill, value=value)


@dataclass(frozen=True)
class MotifAtlasEntry:
    background: str
    size: int
    shape: tuple[tuple[int, int], ...]
    first_divergence_step: int | None
    exact_tail_period: int | None
    dominant_shift: str
    dominant_overlap: float
    final_overlap_with_free_transport: float
    final_population: int
    final_clusters8: int
    final_bbox_area: int
    max_bbox_area: int
    classification: str


@dataclass(frozen=True)
class TripleCollisionEntry:
    background: str
    offsets: tuple[tuple[int, int], tuple[int, int]]
    first_divergence_step: int | None
    exact_tail_period: int | None
    dominant_shift: str
    dominant_overlap: float
    final_overlap_with_free_transport: float
    final_population: int
    final_clusters8: int
    final_bbox_area: int
    max_bbox_area: int
    classification: str


@dataclass(frozen=True)
class RuleObjectAtlas:
    id: int
    stable_index: int
    stable_id: str
    mask: int
    properties: tuple[str, ...]
    tags: tuple[str, ...]
    isolated_particle_velocity: str | None
    isolated_hole_velocity: str | None
    motif_entries: list[MotifAtlasEntry]
    motif_class_counts: dict[str, int]
    triple_entries: list[TripleCollisionEntry]
    triple_class_counts: dict[str, int]


def _classify_entry(
    *,
    first_divergence_step: int | None,
    exact_tail_period: int | None,
    dominant_shift: str,
    dominant_overlap: float,
    final_overlap_with_free_transport: float,
    final_population: int,
    initial_population: int,
    final_clusters8: int,
    final_bbox_area: int,
    max_bbox_area: int,
) -> str:
    if first_divergence_step is None:
        if exact_tail_period == 1 and dominant_shift == "static":
            return "free_fixed"
        if dominant_shift != "static":
            return "free_transporter"
        return "free_periodic"

    if exact_tail_period is not None and final_clusters8 == 1 and final_bbox_area <= 16:
        if dominant_shift == "static":
            return "compact_oscillator"
        return "compact_bound_state"

    if final_population == initial_population and final_clusters8 == final_population and final_bbox_area >= initial_population:
        return "separated_carriers"

    if final_population == initial_population and final_clusters8 >= 2 and final_bbox_area <= 16:
        return "localized_scatter"

    if dominant_overlap > 0.999 and final_overlap_with_free_transport > 0.999:
        return "phase_shift_scatter"

    if max_bbox_area > 64 or final_bbox_area > 25:
        return "broad_scatter"

    return "other_interaction"


def _motif_entries_for_background(
    backend_name: str,
    rule_bits: np.ndarray,
    shapes: dict[int, list[tuple[tuple[int, int], ...]]],
    *,
    background: str,
    velocity: str | None,
) -> list[MotifAtlasEntry]:
    if velocity is None:
        return []
    height = width = 97
    steps = 128
    fill = 0 if background == "zeros" else 1
    value = 1 if background == "zeros" else 0

    entries: list[MotifAtlasEntry] = []
    for size in sorted(shapes):
        for shape in shapes[size]:
            initial = _coords_to_state(height, width, shape, fill=fill, value=value)
            seq = _simulate_sequence(backend_name, rule_bits, initial, steps)
            free_seq = [
                _free_transport_state(height, width, shape, velocity, t, fill=fill, value=value)
                for t in range(steps + 1)
            ]
            first_divergence = None
            shift_names: list[str] = []
            shift_overlaps: list[float] = []
            max_bbox = 0
            for t in range(1, steps + 1):
                if first_divergence is None and not np.array_equal(seq[t], free_seq[t]):
                    first_divergence = t
                shift_name, shift_overlap = _best_shift_reference(seq[t - 1], seq[t])
                shift_names.append(shift_name)
                shift_overlaps.append(shift_overlap)
                active = seq[t] if background == "zeros" else 1 - seq[t]
                max_bbox = max(max_bbox, _bbox_area(active))
            active_final = seq[-1] if background == "zeros" else 1 - seq[-1]
            late_names = shift_names[steps // 2 :]
            dominant_shift = Counter(late_names).most_common(1)[0][0]
            dominant_overlap = float(np.mean(shift_overlaps[steps // 2 :]))
            final_overlap = float(np.mean(seq[-1] == free_seq[-1]))
            final_population = int(active_final.sum())
            final_clusters8 = _cluster_count8(active_final)
            final_bbox = _bbox_area(active_final)
            classification = _classify_entry(
                first_divergence_step=first_divergence,
                exact_tail_period=_detect_tail_period(seq, max_period=32, required_cycles=2),
                dominant_shift=dominant_shift,
                dominant_overlap=dominant_overlap,
                final_overlap_with_free_transport=final_overlap,
                final_population=final_population,
                initial_population=size,
                final_clusters8=final_clusters8,
                final_bbox_area=final_bbox,
                max_bbox_area=max_bbox,
            )
            entries.append(
                MotifAtlasEntry(
                    background=background,
                    size=size,
                    shape=shape,
                    first_divergence_step=first_divergence,
                    exact_tail_period=_detect_tail_period(seq, max_period=32, required_cycles=2),
                    dominant_shift=dominant_shift,
                    dominant_overlap=dominant_overlap,
                    final_overlap_with_free_transport=final_overlap,
                    final_population=final_population,
                    final_clusters8=final_clusters8,
                    final_bbox_area=final_bbox,
                    max_bbox_area=max_bbox,
                    classification=classification,
                )
            )
    return entries


def _triple_entries_for_background(
    backend_name: str,
    rule_bits: np.ndarray,
    *,
    background: str,
    velocity: str | None,
) -> list[TripleCollisionEntry]:
    if velocity is None:
        return []
    fill = 0 if background == "zeros" else 1
    value = 1 if background == "zeros" else 0
    height = width = 129
    steps = 128
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
    ]
    entries: list[TripleCollisionEntry] = []
    for offset_a, offset_b in itertools.combinations(offsets, 2):
        shape = ((0, 0), offset_a, offset_b)
        initial = _coords_to_state(height, width, shape, fill=fill, value=value)
        seq = _simulate_sequence(backend_name, rule_bits, initial, steps)
        free_seq = [
            _free_transport_state(height, width, shape, velocity, t, fill=fill, value=value)
            for t in range(steps + 1)
        ]
        first_divergence = None
        shift_names: list[str] = []
        shift_overlaps: list[float] = []
        max_bbox = 0
        for t in range(1, steps + 1):
            if first_divergence is None and not np.array_equal(seq[t], free_seq[t]):
                first_divergence = t
            shift_name, shift_overlap = _best_shift_reference(seq[t - 1], seq[t])
            shift_names.append(shift_name)
            shift_overlaps.append(shift_overlap)
            active = seq[t] if background == "zeros" else 1 - seq[t]
            max_bbox = max(max_bbox, _bbox_area(active))
        active_final = seq[-1] if background == "zeros" else 1 - seq[-1]
        late_names = shift_names[steps // 2 :]
        dominant_shift = Counter(late_names).most_common(1)[0][0]
        dominant_overlap = float(np.mean(shift_overlaps[steps // 2 :]))
        final_overlap = float(np.mean(seq[-1] == free_seq[-1]))
        final_population = int(active_final.sum())
        final_clusters8 = _cluster_count8(active_final)
        final_bbox = _bbox_area(active_final)
        classification = _classify_entry(
            first_divergence_step=first_divergence,
            exact_tail_period=_detect_tail_period(seq, max_period=32, required_cycles=2),
            dominant_shift=dominant_shift,
            dominant_overlap=dominant_overlap,
            final_overlap_with_free_transport=final_overlap,
            final_population=final_population,
            initial_population=3,
            final_clusters8=final_clusters8,
            final_bbox_area=final_bbox,
            max_bbox_area=max_bbox,
        )
        entries.append(
            TripleCollisionEntry(
                background=background,
                offsets=(offset_a, offset_b),
                first_divergence_step=first_divergence,
                exact_tail_period=_detect_tail_period(seq, max_period=32, required_cycles=2),
                dominant_shift=dominant_shift,
                dominant_overlap=dominant_overlap,
                final_overlap_with_free_transport=final_overlap,
                final_population=final_population,
                final_clusters8=final_clusters8,
                final_bbox_area=final_bbox,
                max_bbox_area=max_bbox,
                classification=classification,
            )
        )
    return entries


def run_study(binary_path: Path, metadata_path: Path, candidate_refs: list[str], backend_name: str, max_size: int) -> dict:
    catalog = load_binary_catalog(binary_path, metadata_path)
    shapes = enumerate_connected_shapes(max_size=max_size, max_span=4)

    reports = []
    for candidate_ref in candidate_refs:
        idx = catalog.resolve_rule_ref(candidate_ref)
        properties = catalog.property_names_for_mask(int(catalog.masks[idx]))
        summary = summarize_simple_rule(catalog.lut_bits[idx].tolist(), properties)
        motif_entries = _motif_entries_for_background(
            backend_name,
            catalog.lut_bits[idx],
            shapes,
            background="zeros",
            velocity=summary.isolated_particle_velocity,
        ) + _motif_entries_for_background(
            backend_name,
            catalog.lut_bits[idx],
            shapes,
            background="ones",
            velocity=summary.isolated_hole_velocity,
        )
        triple_entries = _triple_entries_for_background(
            backend_name,
            catalog.lut_bits[idx],
            background="zeros",
            velocity=summary.isolated_particle_velocity,
        ) + _triple_entries_for_background(
            backend_name,
            catalog.lut_bits[idx],
            background="ones",
            velocity=summary.isolated_hole_velocity,
        )
        reports.append(
            RuleObjectAtlas(
                id=int(catalog.ids[idx]),
                stable_index=int(catalog.stable_indices[idx]),
                stable_id=catalog.stable_ids[idx],
                mask=int(catalog.masks[idx]),
                properties=properties,
                tags=summary.tags,
                isolated_particle_velocity=summary.isolated_particle_velocity,
                isolated_hole_velocity=summary.isolated_hole_velocity,
                motif_entries=motif_entries,
                motif_class_counts=dict(Counter(entry.classification for entry in motif_entries)),
                triple_entries=triple_entries,
                triple_class_counts=dict(Counter(entry.classification for entry in triple_entries)),
            )
        )

    return {
        "binary": str(binary_path),
        "metadata": str(metadata_path),
        "backend": backend_name,
        "candidate_refs": candidate_refs,
        "max_size": max_size,
        "shape_counts": {int(size): len(items) for size, items in shapes.items()},
        "reports": [asdict(report) for report in reports],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Object atlas study for shortlisted CA rules.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--backend", choices=("numpy", "mlx"), default="numpy")
    parser.add_argument("--ids", type=str, nargs="+", required=True)
    parser.add_argument("--max-size", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("object_atlas_study.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    result = run_study(args.binary, args.metadata, args.ids, args.backend, args.max_size)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
