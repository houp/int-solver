from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .binary_catalog import load_binary_catalog
from .simple_filters import summarize_simple_rule
from .simulator import MLXBackend, NumpyBackend, _binary_entropy


DEFAULT_BINARY = Path("expanded_property_panel_nonzero.bin")
DEFAULT_METADATA = Path("expanded_property_panel_nonzero.json")
DEFAULT_EXPLORATION = Path("expanded_property_panel_exploration.json")

SHIFT_OPTIONS = (
    ("static", 0, 0),
    ("north", -1, 0),
    ("south", 1, 0),
    ("east", 0, 1),
    ("west", 0, -1),
    ("northeast", -1, 1),
    ("northwest", -1, -1),
    ("southeast", 1, 1),
    ("southwest", 1, -1),
)


@dataclass(frozen=True)
class ScenarioSummary:
    name: str
    width: int
    height: int
    steps: int
    initial_density: float
    late_mean_activity: float
    late_mean_entropy: float
    late_mean_autocorrelation: float
    late_mean_damage_onebit: float
    late_mean_damage_patch: float
    max_damage_onebit: float
    max_damage_patch: float
    exact_tail_period: int | None
    dominant_shift: str
    dominant_shift_fraction: float
    dominant_shift_overlap: float


@dataclass(frozen=True)
class MotifSummary:
    name: str
    steps: int
    population_series: list[int]
    bbox_area_series: list[int]
    exact_tail_period: int | None
    dominant_shift: str
    dominant_shift_fraction: float


@dataclass(frozen=True)
class CandidateReport:
    id: int
    stable_index: int
    stable_id: str
    mask: int
    properties: tuple[str, ...]
    tags: tuple[str, ...]
    isolated_particle_velocity: str | None
    isolated_hole_velocity: str | None
    scenarios: list[ScenarioSummary]
    motifs: list[MotifSummary]


def _load_candidate_refs(top_k: int, explicit_ids: Iterable[str] | None) -> list[str]:
    if explicit_ids:
        return list(explicit_ids)
    exploration = json.loads(DEFAULT_EXPLORATION.read_text())
    return [str(item.get("stableId") or item["id"]) for item in exploration["top_candidates"][:top_k]]


def _make_random_state(height: int, width: int, probability: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((height, width)) < probability).astype(np.uint8)


def _make_checkerboard(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width))
    return ((x + y) & 1).astype(np.uint8)


def _make_vertical_interface(height: int, width: int) -> np.ndarray:
    state = np.zeros((height, width), dtype=np.uint8)
    state[:, : width // 2] = 1
    return state


def _make_horizontal_stripes(height: int, width: int, stripe_height: int = 4) -> np.ndarray:
    y = np.arange(height)[:, None]
    return (((y // stripe_height) & 1) * np.ones((1, width), dtype=np.uint8)).astype(np.uint8)


def _make_single_particle(height: int, width: int) -> np.ndarray:
    state = np.zeros((height, width), dtype=np.uint8)
    state[height // 2, width // 2] = 1
    return state


def _make_horizontal_pair(height: int, width: int) -> np.ndarray:
    state = np.zeros((height, width), dtype=np.uint8)
    y = height // 2
    x = width // 2
    state[y, x] = 1
    state[y, (x + 1) % width] = 1
    return state


def _make_diagonal_pair(height: int, width: int) -> np.ndarray:
    state = np.zeros((height, width), dtype=np.uint8)
    y = height // 2
    x = width // 2
    state[y, x] = 1
    state[(y + 1) % height, (x + 1) % width] = 1
    return state


def _make_block_2x2(height: int, width: int) -> np.ndarray:
    state = np.zeros((height, width), dtype=np.uint8)
    y = height // 2
    x = width // 2
    state[y : y + 2, x : x + 2] = 1
    return state


def _roll(state: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(state, dy, axis=0), dx, axis=1)


def _detect_tail_period(states: list[np.ndarray], max_period: int = 32, required_cycles: int = 3) -> int | None:
    if len(states) < required_cycles + 1:
        return None
    tail = states[-max(2 * max_period, required_cycles + 1) :]
    for period in range(1, min(max_period, len(tail) - 1) + 1):
        ok = True
        checks = min(required_cycles, len(tail) - period)
        for offset in range(1, checks + 1):
            if not np.array_equal(tail[-offset], tail[-offset - period]):
                ok = False
                break
        if ok:
            return period
    return None


def _dominant_shift_from_pairs(previous_states: list[np.ndarray], next_states: list[np.ndarray]) -> tuple[str, float, float]:
    counts: Counter[str] = Counter()
    overlaps: list[float] = []
    for previous, current in zip(previous_states, next_states):
        best_name = "static"
        best_overlap = -1.0
        for name, dy, dx in SHIFT_OPTIONS:
            overlap = float(np.mean(_roll(previous, dy, dx) == current))
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = name
        counts[best_name] += 1
        overlaps.append(best_overlap)
    dominant_name, dominant_count = counts.most_common(1)[0]
    return dominant_name, dominant_count / len(previous_states), float(np.mean(overlaps))


def _bbox_area(state: np.ndarray) -> int:
    ys, xs = np.nonzero(state)
    if len(xs) == 0:
        return 0
    return int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))


def _create_backend(name: str):
    if name == "mlx":
        return MLXBackend()
    return NumpyBackend()


def _scenario_initial_state(name: str, height: int, width: int, seed: int) -> np.ndarray:
    if name == "random_012":
        return _make_random_state(height, width, 0.12, seed)
    if name == "random_030":
        return _make_random_state(height, width, 0.30, seed)
    if name == "random_050":
        return _make_random_state(height, width, 0.50, seed)
    if name == "checkerboard":
        return _make_checkerboard(height, width)
    if name == "vertical_interface":
        return _make_vertical_interface(height, width)
    if name == "horizontal_stripes":
        return _make_horizontal_stripes(height, width)
    raise ValueError(f"Unknown scenario {name}")


def _run_rule_scenario(backend_name: str, rule_bits: np.ndarray, scenario_name: str, width: int, height: int, steps: int, seed: int) -> ScenarioSummary:
    backend = _create_backend(backend_name)
    if backend_name == "mlx":
        mx = backend.mx
    else:
        mx = None

    initial = _scenario_initial_state(scenario_name, height, width, seed)
    onebit = initial.copy()
    onebit[height // 2, width // 2] ^= 1
    patch = initial.copy()
    patch[height // 2 - 1 : height // 2 + 1, width // 2 - 1 : width // 2 + 1] ^= 1

    states = backend.asarray(initial[None, ...], dtype="uint8")
    onebit_states = backend.asarray(onebit[None, ...], dtype="uint8")
    patch_states = backend.asarray(patch[None, ...], dtype="uint8")
    rules = backend.asarray(rule_bits[None, ...], dtype="uint8")

    density_series: list[float] = []
    entropy_series: list[float] = []
    activity_series: list[float] = []
    autocorr_series: list[float] = []
    damage_onebit_series: list[float] = []
    damage_patch_series: list[float] = []
    saved_states: list[np.ndarray] = [initial.copy()]
    previous_states: list[np.ndarray] = []
    current_states: list[np.ndarray] = []

    prev_np = initial.copy()
    for _ in range(steps):
        states = backend.step_pairwise(states, rules)
        onebit_states = backend.step_pairwise(onebit_states, rules)
        patch_states = backend.step_pairwise(patch_states, rules)

        if backend_name == "mlx":
            current_np = np.asarray(states)[0]
            onebit_np = np.asarray(onebit_states)[0]
            patch_np = np.asarray(patch_states)[0]
        else:
            current_np = np.asarray(states)[0]
            onebit_np = np.asarray(onebit_states)[0]
            patch_np = np.asarray(patch_states)[0]

        density = float(current_np.mean())
        density_series.append(density)
        entropy_series.append(_binary_entropy(density))
        activity_series.append(float(np.mean(current_np != prev_np)))
        autocorr_series.append(float(np.mean(current_np == initial)))
        damage_onebit_series.append(float(np.mean(current_np != onebit_np)))
        damage_patch_series.append(float(np.mean(current_np != patch_np)))
        previous_states.append(prev_np.copy())
        current_states.append(current_np.copy())
        saved_states.append(current_np.copy())
        prev_np = current_np

    late_slice = slice(steps // 2, None)
    dominant_shift, dominant_fraction, dominant_overlap = _dominant_shift_from_pairs(
        previous_states[late_slice], current_states[late_slice]
    )
    return ScenarioSummary(
        name=scenario_name,
        width=width,
        height=height,
        steps=steps,
        initial_density=float(initial.mean()),
        late_mean_activity=float(np.mean(activity_series[late_slice])),
        late_mean_entropy=float(np.mean(entropy_series[late_slice])),
        late_mean_autocorrelation=float(np.mean(autocorr_series[late_slice])),
        late_mean_damage_onebit=float(np.mean(damage_onebit_series[late_slice])),
        late_mean_damage_patch=float(np.mean(damage_patch_series[late_slice])),
        max_damage_onebit=float(np.max(damage_onebit_series)),
        max_damage_patch=float(np.max(damage_patch_series)),
        exact_tail_period=_detect_tail_period(saved_states),
        dominant_shift=dominant_shift,
        dominant_shift_fraction=dominant_fraction,
        dominant_shift_overlap=dominant_overlap,
    )


def _run_rule_motif(backend_name: str, rule_bits: np.ndarray, motif_name: str, width: int, height: int, steps: int) -> MotifSummary:
    motif_builders = {
        "single_particle": _make_single_particle,
        "horizontal_pair": _make_horizontal_pair,
        "diagonal_pair": _make_diagonal_pair,
        "block_2x2": _make_block_2x2,
    }
    initial = motif_builders[motif_name](height, width)
    backend = _create_backend(backend_name)
    states = backend.asarray(initial[None, ...], dtype="uint8")
    rules = backend.asarray(rule_bits[None, ...], dtype="uint8")

    saved_states = [initial.copy()]
    population_series = [int(initial.sum())]
    bbox_area_series = [_bbox_area(initial)]
    previous_states: list[np.ndarray] = []
    current_states: list[np.ndarray] = []
    prev_np = initial.copy()

    for _ in range(steps):
        states = backend.step_pairwise(states, rules)
        current_np = np.asarray(states)[0]
        population_series.append(int(current_np.sum()))
        bbox_area_series.append(_bbox_area(current_np))
        previous_states.append(prev_np.copy())
        current_states.append(current_np.copy())
        saved_states.append(current_np.copy())
        prev_np = current_np

    dominant_shift, dominant_fraction, _ = _dominant_shift_from_pairs(previous_states, current_states)
    return MotifSummary(
        name=motif_name,
        steps=steps,
        population_series=population_series,
        bbox_area_series=bbox_area_series,
        exact_tail_period=_detect_tail_period(saved_states, max_period=16, required_cycles=2),
        dominant_shift=dominant_shift,
        dominant_shift_fraction=dominant_fraction,
    )


def analyze_candidates(
    binary_path: Path,
    metadata_path: Path,
    candidate_refs: list[str],
    backend: str,
) -> dict:
    catalog = load_binary_catalog(binary_path, metadata_path)

    scenario_specs = [
        ("random_012", 128, 128, 256, 1012),
        ("random_030", 128, 128, 256, 1030),
        ("random_050", 128, 128, 256, 1050),
        ("checkerboard", 128, 128, 192, 0),
        ("vertical_interface", 128, 128, 192, 0),
        ("horizontal_stripes", 128, 128, 192, 0),
    ]
    motif_specs = [
        ("single_particle", 33, 33, 24),
        ("horizontal_pair", 33, 33, 24),
        ("diagonal_pair", 33, 33, 24),
        ("block_2x2", 33, 33, 24),
    ]

    reports: list[CandidateReport] = []
    for candidate_ref in candidate_refs:
        index = catalog.resolve_rule_ref(candidate_ref)
        rule_bits = catalog.lut_bits[index]
        summary = summarize_simple_rule(rule_bits.tolist(), catalog.property_names_for_mask(int(catalog.masks[index])))
        scenarios = [
            _run_rule_scenario(backend, rule_bits, name, width, height, steps, seed)
            for name, width, height, steps, seed in scenario_specs
        ]
        motifs = [
            _run_rule_motif(backend, rule_bits, name, width, height, steps)
            for name, width, height, steps in motif_specs
        ]
        reports.append(
            CandidateReport(
                id=int(catalog.ids[index]),
                stable_index=int(catalog.stable_indices[index]),
                stable_id=catalog.stable_ids[index],
                mask=int(catalog.masks[index]),
                properties=catalog.property_names_for_mask(int(catalog.masks[index])),
                tags=summary.tags,
                isolated_particle_velocity=summary.isolated_particle_velocity,
                isolated_hole_velocity=summary.isolated_hole_velocity,
                scenarios=scenarios,
                motifs=motifs,
            )
        )

    return {
        "binary": str(binary_path),
        "metadata": str(metadata_path),
        "backend": backend,
        "candidate_refs": candidate_refs,
        "reports": [asdict(report) for report in reports],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deeper analysis on selected candidate rules.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--backend", choices=("numpy", "mlx"), default="numpy")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ids", type=str, nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("candidate_deep_analysis.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    candidate_refs = _load_candidate_refs(args.top_k, args.ids)
    result = analyze_candidates(
        binary_path=args.binary,
        metadata_path=args.metadata,
        candidate_refs=candidate_refs,
        backend=args.backend,
    )
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
