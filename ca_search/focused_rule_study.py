from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .binary_catalog import load_binary_catalog
from .simple_filters import summarize_simple_rule
from .simulator import MLXBackend, NumpyBackend, _binary_entropy


DEFAULT_BINARY = Path("expanded_property_panel_nonzero.bin")
DEFAULT_METADATA = Path("expanded_property_panel_nonzero.json")

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
SHIFT_BY_NAME = {name: (dy, dx) for name, dy, dx in SHIFT_OPTIONS}


@dataclass(frozen=True)
class EnsembleSummary:
    name: str
    probability: float
    seeds: list[int]
    width: int
    height: int
    steps: int
    late_mean_activity: float
    late_mean_entropy: float
    late_mean_damage_onebit: float
    late_mean_damage_patch: float
    max_damage_onebit: float
    max_damage_patch: float
    threshold_step_damage_onebit_001: int | None
    threshold_step_damage_onebit_005: int | None
    threshold_step_damage_patch_001: int | None
    threshold_step_damage_patch_005: int | None
    dominant_shift: str
    dominant_shift_fraction: float
    dominant_shift_overlap: float
    moving_period_1_fraction: float
    moving_period_2_fraction: float
    moving_period_4_fraction: float


@dataclass(frozen=True)
class StructuredSummary:
    name: str
    width: int
    height: int
    steps: int
    late_mean_activity: float
    late_mean_entropy: float
    exact_tail_period: int | None
    dominant_shift: str
    dominant_shift_fraction: float
    dominant_shift_overlap: float
    moving_period_1: bool
    moving_period_2: bool
    moving_period_4: bool


@dataclass(frozen=True)
class MotifTrajectory:
    name: str
    background: str
    steps: int
    population_series: list[int]
    bbox_area_series: list[int]
    exact_tail_period: int | None
    dominant_shift: str
    dominant_shift_fraction: float
    moving_period_1: bool
    moving_period_2: bool
    moving_period_4: bool


@dataclass(frozen=True)
class RuleStudy:
    id: int
    stable_index: int
    stable_id: str
    mask: int
    properties: tuple[str, ...]
    tags: tuple[str, ...]
    isolated_particle_velocity: str | None
    isolated_hole_velocity: str | None
    ensembles: list[EnsembleSummary]
    structured: list[StructuredSummary]
    motifs: list[MotifTrajectory]


def _backend(name: str):
    if name == "mlx":
        return MLXBackend()
    return NumpyBackend()


def _make_random_state(height: int, width: int, probability: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((height, width)) < probability).astype(np.uint8)


def _roll(state: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(state, dy, axis=0), dx, axis=1)


def _bbox_area(state: np.ndarray) -> int:
    ys, xs = np.nonzero(state)
    if len(xs) == 0:
        return 0
    return int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))


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


def _dominant_shift(previous_states: list[np.ndarray], current_states: list[np.ndarray]) -> tuple[str, float, float]:
    counts: Counter[str] = Counter()
    overlaps: list[float] = []
    for previous, current in zip(previous_states, current_states):
        best_name = "static"
        best_overlap = -1.0
        for name, dy, dx in SHIFT_OPTIONS:
            overlap = float(np.mean(_roll(previous, dy, dx) == current))
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = name
        counts[best_name] += 1
        overlaps.append(best_overlap)
    name, count = counts.most_common(1)[0]
    return name, count / len(previous_states), float(np.mean(overlaps))


def _moving_period_fraction(states: list[np.ndarray], shift_name: str, period: int) -> float:
    dy, dx = SHIFT_BY_NAME[shift_name]
    matches = 0
    total = 0
    for t in range(period, len(states)):
        total += 1
        if np.array_equal(states[t], _roll(states[t - period], period * dy, period * dx)):
            matches += 1
    if total == 0:
        return 0.0
    return matches / total


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


def _make_diagonal_stripes(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width))
    return (((x + y) // 4) & 1).astype(np.uint8)


def _make_single_particle(height: int, width: int) -> np.ndarray:
    state = np.zeros((height, width), dtype=np.uint8)
    state[height // 2, width // 2] = 1
    return state


def _make_single_hole(height: int, width: int) -> np.ndarray:
    state = np.ones((height, width), dtype=np.uint8)
    state[height // 2, width // 2] = 0
    return state


def _make_block_2x2(height: int, width: int, value: int = 1) -> np.ndarray:
    state = np.zeros((height, width), dtype=np.uint8) if value else np.ones((height, width), dtype=np.uint8)
    y = height // 2
    x = width // 2
    state[y : y + 2, x : x + 2] = value
    return state


def _make_l_shape(height: int, width: int) -> np.ndarray:
    state = np.zeros((height, width), dtype=np.uint8)
    y = height // 2
    x = width // 2
    state[y, x] = 1
    state[y + 1, x] = 1
    state[y + 1, x + 1] = 1
    return state


def _run_ensemble(
    backend_name: str,
    rule_bits: np.ndarray,
    probability: float,
    seeds: list[int],
    width: int,
    height: int,
    steps: int,
    name: str,
) -> EnsembleSummary:
    backend = _backend(backend_name)
    rule = backend.asarray(rule_bits[None, ...], dtype="uint8")

    late_activities = []
    late_entropies = []
    late_damage_onebit = []
    late_damage_patch = []
    max_damage_onebit = []
    max_damage_patch = []
    threshold_onebit_001 = []
    threshold_onebit_005 = []
    threshold_patch_001 = []
    threshold_patch_005 = []
    dominant_counter: Counter[str] = Counter()
    dominant_overlaps = []
    moving1 = []
    moving2 = []
    moving4 = []

    for seed in seeds:
        initial = _make_random_state(height, width, probability, seed)
        onebit = initial.copy()
        onebit[height // 2, width // 2] ^= 1
        patch = initial.copy()
        patch[height // 2 - 1 : height // 2 + 1, width // 2 - 1 : width // 2 + 1] ^= 1

        states = backend.asarray(initial[None, ...], dtype="uint8")
        states_onebit = backend.asarray(onebit[None, ...], dtype="uint8")
        states_patch = backend.asarray(patch[None, ...], dtype="uint8")

        prev = initial.copy()
        saved_states = [initial.copy()]
        previous_states = []
        current_states = []
        activity_series = []
        entropy_series = []
        damage_onebit_series = []
        damage_patch_series = []

        for _ in range(steps):
            states = backend.step_pairwise(states, rule)
            states_onebit = backend.step_pairwise(states_onebit, rule)
            states_patch = backend.step_pairwise(states_patch, rule)

            current = np.asarray(states)[0]
            current_onebit = np.asarray(states_onebit)[0]
            current_patch = np.asarray(states_patch)[0]

            activity_series.append(float(np.mean(current != prev)))
            density = float(current.mean())
            entropy_series.append(_binary_entropy(density))
            damage_onebit_series.append(float(np.mean(current != current_onebit)))
            damage_patch_series.append(float(np.mean(current != current_patch)))
            previous_states.append(prev.copy())
            current_states.append(current.copy())
            saved_states.append(current.copy())
            prev = current

        late = slice(steps // 2, None)
        late_activities.append(float(np.mean(activity_series[late])))
        late_entropies.append(float(np.mean(entropy_series[late])))
        late_damage_onebit.append(float(np.mean(damage_onebit_series[late])))
        late_damage_patch.append(float(np.mean(damage_patch_series[late])))
        max_damage_onebit.append(float(np.max(damage_onebit_series)))
        max_damage_patch.append(float(np.max(damage_patch_series)))

        def first_crossing(series: list[float], threshold: float):
            for i, value in enumerate(series):
                if value >= threshold:
                    return i + 1
            return None

        threshold_onebit_001.append(first_crossing(damage_onebit_series, 0.01))
        threshold_onebit_005.append(first_crossing(damage_onebit_series, 0.05))
        threshold_patch_001.append(first_crossing(damage_patch_series, 0.01))
        threshold_patch_005.append(first_crossing(damage_patch_series, 0.05))

        shift_name, _, shift_overlap = _dominant_shift(previous_states[late], current_states[late])
        dominant_counter[shift_name] += 1
        dominant_overlaps.append(shift_overlap)
        moving1.append(_moving_period_fraction(saved_states, shift_name, 1))
        moving2.append(_moving_period_fraction(saved_states, shift_name, 2))
        moving4.append(_moving_period_fraction(saved_states, shift_name, 4))

    dominant_name, dominant_count = dominant_counter.most_common(1)[0]

    def _mean_optional(values: list[int | None]) -> int | None:
        filtered = [v for v in values if v is not None]
        if not filtered:
            return None
        return int(round(sum(filtered) / len(filtered)))

    return EnsembleSummary(
        name=name,
        probability=probability,
        seeds=seeds,
        width=width,
        height=height,
        steps=steps,
        late_mean_activity=float(np.mean(late_activities)),
        late_mean_entropy=float(np.mean(late_entropies)),
        late_mean_damage_onebit=float(np.mean(late_damage_onebit)),
        late_mean_damage_patch=float(np.mean(late_damage_patch)),
        max_damage_onebit=float(np.max(max_damage_onebit)),
        max_damage_patch=float(np.max(max_damage_patch)),
        threshold_step_damage_onebit_001=_mean_optional(threshold_onebit_001),
        threshold_step_damage_onebit_005=_mean_optional(threshold_onebit_005),
        threshold_step_damage_patch_001=_mean_optional(threshold_patch_001),
        threshold_step_damage_patch_005=_mean_optional(threshold_patch_005),
        dominant_shift=dominant_name,
        dominant_shift_fraction=dominant_count / len(seeds),
        dominant_shift_overlap=float(np.mean(dominant_overlaps)),
        moving_period_1_fraction=float(np.mean(moving1)),
        moving_period_2_fraction=float(np.mean(moving2)),
        moving_period_4_fraction=float(np.mean(moving4)),
    )


def _run_structured(
    backend_name: str,
    rule_bits: np.ndarray,
    name: str,
    initial: np.ndarray,
    steps: int,
) -> StructuredSummary:
    backend = _backend(backend_name)
    rule = backend.asarray(rule_bits[None, ...], dtype="uint8")
    states = backend.asarray(initial[None, ...], dtype="uint8")
    prev = initial.copy()
    saved_states = [initial.copy()]
    previous_states = []
    current_states = []
    activity_series = []
    entropy_series = []

    for _ in range(steps):
        states = backend.step_pairwise(states, rule)
        current = np.asarray(states)[0]
        activity_series.append(float(np.mean(current != prev)))
        entropy_series.append(_binary_entropy(float(current.mean())))
        previous_states.append(prev.copy())
        current_states.append(current.copy())
        saved_states.append(current.copy())
        prev = current

    late = slice(steps // 2, None)
    shift_name, shift_fraction, shift_overlap = _dominant_shift(previous_states[late], current_states[late])
    return StructuredSummary(
        name=name,
        width=initial.shape[1],
        height=initial.shape[0],
        steps=steps,
        late_mean_activity=float(np.mean(activity_series[late])),
        late_mean_entropy=float(np.mean(entropy_series[late])),
        exact_tail_period=_detect_tail_period(saved_states),
        dominant_shift=shift_name,
        dominant_shift_fraction=shift_fraction,
        dominant_shift_overlap=shift_overlap,
        moving_period_1=_moving_period_fraction(saved_states, shift_name, 1) == 1.0,
        moving_period_2=_moving_period_fraction(saved_states, shift_name, 2) == 1.0,
        moving_period_4=_moving_period_fraction(saved_states, shift_name, 4) == 1.0,
    )


def _run_motif(
    backend_name: str,
    rule_bits: np.ndarray,
    name: str,
    initial: np.ndarray,
    steps: int,
    background: str,
) -> MotifTrajectory:
    backend = _backend(backend_name)
    rule = backend.asarray(rule_bits[None, ...], dtype="uint8")
    states = backend.asarray(initial[None, ...], dtype="uint8")
    prev = initial.copy()
    saved_states = [initial.copy()]
    previous_states = []
    current_states = []
    population_series = [int(initial.sum())]
    bbox_area_series = [_bbox_area(initial)]

    for _ in range(steps):
        states = backend.step_pairwise(states, rule)
        current = np.asarray(states)[0]
        population_series.append(int(current.sum()))
        bbox_area_series.append(_bbox_area(current))
        previous_states.append(prev.copy())
        current_states.append(current.copy())
        saved_states.append(current.copy())
        prev = current

    shift_name, shift_fraction, _ = _dominant_shift(previous_states, current_states)
    return MotifTrajectory(
        name=name,
        background=background,
        steps=steps,
        population_series=population_series,
        bbox_area_series=bbox_area_series,
        exact_tail_period=_detect_tail_period(saved_states, max_period=16, required_cycles=2),
        dominant_shift=shift_name,
        dominant_shift_fraction=shift_fraction,
        moving_period_1=_moving_period_fraction(saved_states, shift_name, 1) == 1.0,
        moving_period_2=_moving_period_fraction(saved_states, shift_name, 2) == 1.0,
        moving_period_4=_moving_period_fraction(saved_states, shift_name, 4) == 1.0,
    )


def analyze_rule(backend_name: str, rule_bits: np.ndarray) -> tuple[list[EnsembleSummary], list[StructuredSummary], list[MotifTrajectory]]:
    ensembles = [
        _run_ensemble(backend_name, rule_bits, 0.05, [105, 205, 305, 405], 256, 256, 512, "random_005"),
        _run_ensemble(backend_name, rule_bits, 0.12, [112, 212, 312, 412], 256, 256, 512, "random_012"),
        _run_ensemble(backend_name, rule_bits, 0.30, [130, 230, 330, 430], 256, 256, 512, "random_030"),
        _run_ensemble(backend_name, rule_bits, 0.50, [150, 250, 350, 450], 256, 256, 512, "random_050"),
    ]

    structured = [
        _run_structured(backend_name, rule_bits, "checkerboard", _make_checkerboard(256, 256), 256),
        _run_structured(backend_name, rule_bits, "vertical_interface", _make_vertical_interface(256, 256), 256),
        _run_structured(backend_name, rule_bits, "horizontal_stripes", _make_horizontal_stripes(256, 256), 256),
        _run_structured(backend_name, rule_bits, "diagonal_stripes", _make_diagonal_stripes(256, 256), 256),
    ]

    motifs = [
        _run_motif(backend_name, rule_bits, "single_particle", _make_single_particle(65, 65), 64, "zeros"),
        _run_motif(backend_name, rule_bits, "block_2x2", _make_block_2x2(65, 65, 1), 64, "zeros"),
        _run_motif(backend_name, rule_bits, "l_shape", _make_l_shape(65, 65), 64, "zeros"),
        _run_motif(backend_name, rule_bits, "single_hole", _make_single_hole(65, 65), 64, "ones"),
        _run_motif(backend_name, rule_bits, "hole_block_2x2", _make_block_2x2(65, 65, 0), 64, "ones"),
    ]
    return ensembles, structured, motifs


def run_study(binary_path: Path, metadata_path: Path, candidate_refs: list[str], backend_name: str) -> dict:
    catalog = load_binary_catalog(binary_path, metadata_path)
    reports = []
    for candidate_ref in candidate_refs:
        idx = catalog.resolve_rule_ref(candidate_ref)
        summary = summarize_simple_rule(
            catalog.lut_bits[idx].tolist(),
            catalog.property_names_for_mask(int(catalog.masks[idx])),
        )
        ensembles, structured, motifs = analyze_rule(backend_name, catalog.lut_bits[idx])
        reports.append(
            RuleStudy(
                id=int(catalog.ids[idx]),
                stable_index=int(catalog.stable_indices[idx]),
                stable_id=catalog.stable_ids[idx],
                mask=int(catalog.masks[idx]),
                properties=catalog.property_names_for_mask(int(catalog.masks[idx])),
                tags=summary.tags,
                isolated_particle_velocity=summary.isolated_particle_velocity,
                isolated_hole_velocity=summary.isolated_hole_velocity,
                ensembles=ensembles,
                structured=structured,
                motifs=motifs,
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
    parser = argparse.ArgumentParser(description="Focused study of selected candidate rules.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--backend", choices=("numpy", "mlx"), default="numpy")
    parser.add_argument("--ids", type=str, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("focused_rule_study.json"))
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
