from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .binary_catalog import BinaryCatalog, load_binary_catalog
from .lut import (
    VELOCITY_TO_SOURCE_POSITION,
    rule_stable_id,
    rigid_rule_bits,
)
from .simulator import create_backend


@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    phases: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ScheduleEvaluation:
    schedule: str
    grid: str
    trials: int
    tied_initial_states: int
    effective_trials: int
    preprocess_steps: int
    final_steps: int
    local_majority_agreement: float
    local_density_variance: float
    preprocess_checkerboard_alignment: float
    preprocess_orthogonal_disagreement: float
    preprocess_checkerboard_2x2_fraction: float
    final_majority_accuracy: float
    final_consensus_accuracy: float
    final_consensus_rate: float
    mean_final_density: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessorEvaluation:
    legacy_index: int
    stable_index: int
    stable_id: str
    schedule_name: str
    preprocess_steps: int
    probability: float
    trials: int
    local_majority_agreement: float
    local_density_variance: float
    isolated_particle_velocity: str | None
    isolated_hole_velocity: str | None
    mask: int
    ones: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BalancedPreprocessorEvaluation:
    stable_index: int
    stable_id: str
    preprocess_steps: int
    final_steps: int
    probabilities: tuple[float, ...]
    trials: int
    min_local_majority_agreement: float
    max_local_density_variance: float
    min_final_majority_accuracy: float
    mean_final_majority_accuracy: float
    final_majority_accuracy_gap: float
    min_final_consensus_rate: float
    isolated_particle_velocity: str | None
    isolated_hole_velocity: str | None
    mask: int
    ones: int
    per_probability: tuple[dict[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RepeatedBlockEvaluation:
    stable_index: int
    stable_id: str
    preprocess_steps: int
    amplifier_steps: int
    repetitions: int
    amplifier_name: str
    probabilities: tuple[float, ...]
    trials: int
    min_final_consensus_rate: float
    min_final_consensus_accuracy: float
    min_final_majority_accuracy: float
    mean_final_majority_accuracy: float
    final_majority_accuracy_gap: float
    max_local_density_variance: float
    min_local_majority_agreement: float
    isolated_particle_velocity: str | None
    isolated_hole_velocity: str | None
    mask: int
    ones: int
    per_probability: tuple[dict[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FiniteSwitchEvaluation:
    stable_index: int
    stable_id: str
    preprocess_steps: int
    amplifier_steps: int
    amplifier_name: str
    probabilities: tuple[float, ...]
    trials: int
    min_final_consensus_rate: float
    min_final_consensus_accuracy: float
    min_final_majority_accuracy: float
    mean_final_majority_accuracy: float
    final_majority_accuracy_gap: float
    max_local_density_variance: float
    min_local_majority_agreement: float
    isolated_particle_velocity: str | None
    isolated_hole_velocity: str | None
    mask: int
    ones: int
    per_probability: tuple[dict[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


TRIVIAL_STABLE_IDS = frozenset(
    rule_stable_id(rigid_rule_bits(position))
    for position in ("u", "t", "w", "y", "b", "a", "c", "x", "z")
)


def embedded_diagonal_traffic_rule_bits(velocity: str) -> list[int]:
    if velocity not in {"northeast", "northwest", "southeast", "southwest"}:
        raise ValueError(f"Unsupported diagonal traffic velocity: {velocity}")

    source_lookup = {
        "northeast": "a",
        "northwest": "c",
        "southeast": "x",
        "southwest": "z",
    }
    destination_lookup = {
        "northeast": "z",
        "northwest": "x",
        "southeast": "c",
        "southwest": "a",
    }
    source = source_lookup[velocity]
    destination = destination_lookup[velocity]

    pos_to_shift = {
        "x": 8,
        "y": 7,
        "z": 6,
        "t": 5,
        "u": 4,
        "w": 3,
        "a": 2,
        "b": 1,
        "c": 0,
    }

    bits = [0] * 512
    for index in range(512):
        source_value = (index >> pos_to_shift[source]) & 1
        center = (index >> 4) & 1
        destination_value = (index >> pos_to_shift[destination]) & 1
        bits[index] = int((source_value and not center) or (center and destination_value))
    return bits


def moore_majority_rule_bits() -> list[int]:
    bits = [0] * 512
    for index in range(512):
        bits[index] = 1 if index.bit_count() >= 5 else 0
    return bits


def moore_threshold_rule_bits(threshold: int) -> list[int]:
    if not 0 <= threshold <= 9:
        raise ValueError("Moore threshold must be in [0, 9]")
    bits = [0] * 512
    for index in range(512):
        bits[index] = 1 if index.bit_count() >= threshold else 0
    return bits


def von_neumann_majority_rule_bits() -> list[int]:
    bits = [0] * 512
    for index in range(512):
        y = (index >> 7) & 1
        t = (index >> 5) & 1
        u = (index >> 4) & 1
        w = (index >> 3) & 1
        b = (index >> 1) & 1
        bits[index] = 1 if (y + t + u + w + b) >= 3 else 0
    return bits


def von_neumann_threshold_rule_bits(threshold: int) -> list[int]:
    if not 0 <= threshold <= 5:
        raise ValueError("von Neumann threshold must be in [0, 5]")
    bits = [0] * 512
    for index in range(512):
        y = (index >> 7) & 1
        t = (index >> 5) & 1
        u = (index >> 4) & 1
        w = (index >> 3) & 1
        b = (index >> 1) & 1
        bits[index] = 1 if (y + t + u + w + b) >= threshold else 0
    return bits


def load_named_rules(binary_path: str, metadata_path: str, selectors: dict[str, str]) -> dict[str, list[int]]:
    catalog = load_binary_catalog(binary_path, metadata_path)
    out: dict[str, list[int]] = {}
    for name, selector in selectors.items():
        index = catalog.resolve_rule_ref(selector)
        out[name] = catalog.lut_bits[index].tolist()
    return out


def evaluate_schedule_suite(
    backend_name: str,
    schedules: Iterable[ScheduleSpec],
    rule_bank: dict[str, list[int]],
    *,
    width: int,
    height: int,
    probabilities: Iterable[float],
    trials_per_probability: int,
    seed: int,
    majority_metric: str = "moore",
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for probability in probabilities:
        initial_states = (
            rng.random((trials_per_probability, height, width)) < float(probability)
        ).astype(np.uint8)
        for schedule in schedules:
            report = evaluate_schedule(
                backend_name,
                schedule,
                rule_bank,
                initial_states,
                majority_metric=majority_metric,
            )
            reports.append(
                {
                    "probability": float(probability),
                    "schedule": report.to_dict(),
                }
            )
    return reports


def evaluate_schedule(
    backend_name: str,
    schedule: ScheduleSpec,
    rule_bank: dict[str, list[int]],
    initial_states: np.ndarray,
    *,
    majority_metric: str = "moore",
) -> ScheduleEvaluation:
    if initial_states.ndim != 3:
        raise ValueError("initial_states must have shape (batch, height, width)")

    backend = create_backend(backend_name)
    states = backend.asarray(initial_states, dtype="uint8")
    batch, height, width = initial_states.shape

    trial_majorities = initial_states.reshape(batch, -1).sum(axis=1)
    total_sites = width * height
    initial_labels = np.where(trial_majorities > (total_sites / 2), 1, 0)
    tied_mask = trial_majorities * 2 == total_sites
    effective_trials = int((~tied_mask).sum())

    preprocess_steps = sum(steps for _, steps in schedule.phases[:-1])
    final_steps = schedule.phases[-1][1] if schedule.phases else 0

    preprocess_snapshot = backend.to_numpy(states)
    for phase_index, (rule_name, steps) in enumerate(schedule.phases):
        batched_rule = _tile_rule_for_batch(rule_bank[rule_name], batch, backend_name)
        for _ in range(steps):
            states = backend.step_pairwise(states, batched_rule)
        if phase_index == len(schedule.phases) - 2:
            preprocess_snapshot = backend.to_numpy(states)

    final_states = backend.to_numpy(states)
    if len(schedule.phases) <= 1:
        preprocess_snapshot = np.asarray(initial_states, dtype=np.uint8).copy()

    local_majority_agreement, local_density_variance = _measure_preprocess_quality(
        preprocess_snapshot, initial_labels, tied_mask, majority_metric
    )
    (
        preprocess_checkerboard_alignment,
        preprocess_orthogonal_disagreement,
        preprocess_checkerboard_2x2_fraction,
    ) = _measure_checkerboard_structure(preprocess_snapshot)
    final_majority_accuracy, final_consensus_accuracy, final_consensus_rate, mean_final_density = (
        _measure_final_classification(final_states, initial_labels, tied_mask)
    )

    return ScheduleEvaluation(
        schedule=schedule.name,
        grid=f"{width}x{height}",
        trials=batch,
        tied_initial_states=int(tied_mask.sum()),
        effective_trials=effective_trials,
        preprocess_steps=preprocess_steps,
        final_steps=final_steps,
        local_majority_agreement=local_majority_agreement,
        local_density_variance=local_density_variance,
        preprocess_checkerboard_alignment=preprocess_checkerboard_alignment,
        preprocess_orthogonal_disagreement=preprocess_orthogonal_disagreement,
        preprocess_checkerboard_2x2_fraction=preprocess_checkerboard_2x2_fraction,
        final_majority_accuracy=final_majority_accuracy,
        final_consensus_accuracy=final_consensus_accuracy,
        final_consensus_rate=final_consensus_rate,
        mean_final_density=mean_final_density,
    )


def default_density_rule_bank(binary_path: str, metadata_path: str) -> dict[str, list[int]]:
    selectors = {
        "cand_029b09cea": "sid:029b09cea0b5",
        "cand_812b7dae7": "sid:812b7dae7aa7",
        "cand_f076fc6ff": "sid:f076fc6ffb58",
        "cand_adb15d422": "sid:adb15d42281e",
        "cand_ca6705f68": "sid:ca6705f68694",
        "cand_58ed6b657": "sid:58ed6b657afb",
        "cand_b9b31aebc": "sid:b9b31aebc9b5",
        "cand_6774f7e43": "sid:6774f7e4355e",
        "cand_3fd35ca10": "sid:3fd35ca10c2e",
        "cand_13d4fec69": "sid:13d4fec69618",
        "cand_fdabfae81": "sid:fdabfae81f3f",
    }
    bank = load_named_rules(binary_path, metadata_path, selectors)
    bank["majority_moore"] = moore_majority_rule_bits()
    bank["majority_vn"] = von_neumann_majority_rule_bits()

    cardinal_source = {
        "shift_north": "b",
        "shift_south": "y",
        "shift_east": "t",
        "shift_west": "w",
    }
    for name, position in cardinal_source.items():
        bank[name] = rigid_rule_bits(position)

    # Embedded 1D traffic rules inside the von Neumann cross.
    bank["traffic_east"] = _embedded_cardinal_traffic("east")
    bank["traffic_west"] = _embedded_cardinal_traffic("west")
    bank["traffic_north"] = _embedded_cardinal_traffic("north")
    bank["traffic_south"] = _embedded_cardinal_traffic("south")

    bank["diag_traffic_ne"] = embedded_diagonal_traffic_rule_bits("northeast")
    bank["diag_traffic_nw"] = embedded_diagonal_traffic_rule_bits("northwest")
    bank["diag_traffic_se"] = embedded_diagonal_traffic_rule_bits("southeast")
    bank["diag_traffic_sw"] = embedded_diagonal_traffic_rule_bits("southwest")
    return bank


def default_density_schedules(
    *,
    preprocess_short: int = 32,
    preprocess_long: int = 64,
    majority_tail: int = 32,
    majority_name: str = "majority_moore",
) -> list[ScheduleSpec]:
    return [
        ScheduleSpec("majority_only", ((majority_name, majority_tail),)),
        ScheduleSpec(
            "traffic_cardinal_cycle_short_then_majority",
            (
                ("traffic_east", preprocess_short // 4),
                ("traffic_north", preprocess_short // 4),
                ("traffic_west", preprocess_short // 4),
                ("traffic_south", preprocess_short // 4),
                (majority_name, majority_tail),
            ),
        ),
        ScheduleSpec(
            "traffic_cardinal_cycle_long_then_majority",
            (
                ("traffic_east", preprocess_long // 4),
                ("traffic_north", preprocess_long // 4),
                ("traffic_west", preprocess_long // 4),
                ("traffic_south", preprocess_long // 4),
                (majority_name, majority_tail),
            ),
        ),
        ScheduleSpec(
            "traffic_diagonal_cycle_short_then_majority",
            (
                ("diag_traffic_ne", preprocess_short // 4),
                ("diag_traffic_nw", preprocess_short // 4),
                ("diag_traffic_sw", preprocess_short // 4),
                ("diag_traffic_se", preprocess_short // 4),
                (majority_name, majority_tail),
            ),
        ),
        ScheduleSpec(
            "traffic_diagonal_cycle_long_then_majority",
            (
                ("diag_traffic_ne", preprocess_long // 4),
                ("diag_traffic_nw", preprocess_long // 4),
                ("diag_traffic_sw", preprocess_long // 4),
                ("diag_traffic_se", preprocess_long // 4),
                (majority_name, majority_tail),
            ),
        ),
        ScheduleSpec(
            "candidate_029b_short_then_majority",
            (("cand_029b09cea", preprocess_short), (majority_name, majority_tail)),
        ),
        ScheduleSpec(
            "candidate_812b_short_then_majority",
            (("cand_812b7dae7", preprocess_short), (majority_name, majority_tail)),
        ),
        ScheduleSpec(
            "candidate_f076_short_then_majority",
            (("cand_f076fc6ff", preprocess_short), (majority_name, majority_tail)),
        ),
        ScheduleSpec(
            "candidate_029b_long_then_majority",
            (("cand_029b09cea", preprocess_long), (majority_name, majority_tail)),
        ),
        ScheduleSpec(
            "candidate_812b_long_then_majority",
            (("cand_812b7dae7", preprocess_long), (majority_name, majority_tail)),
        ),
    ]


def screen_preprocessors(
    catalog: BinaryCatalog,
    *,
    backend_name: str,
    width: int,
    height: int,
    probability: float,
    trials: int,
    seed: int,
    preprocess_steps: int,
    max_rules: int | None = None,
    legacy_indices: Iterable[int] | None = None,
    require_nontrivial: bool = True,
    rule_batch_size: int = 64,
    majority_metric: str = "moore",
    top_k: int = 25,
) -> list[PreprocessorEvaluation]:
    rng = np.random.default_rng(seed)
    initial_states = (rng.random((trials, height, width)) < float(probability)).astype(np.uint8)
    trial_majorities = initial_states.reshape(trials, -1).sum(axis=1)
    total_sites = width * height
    initial_labels = np.where(trial_majorities > (total_sites / 2), 1, 0)
    tied_mask = trial_majorities * 2 == total_sites

    selected_indices = _resolve_catalog_indices(
        catalog,
        max_rules=max_rules,
        legacy_indices=legacy_indices,
        require_nontrivial=require_nontrivial,
    )
    if not selected_indices:
        return []

    backend = create_backend(backend_name)
    top_results: list[PreprocessorEvaluation] = []

    for start in range(0, len(selected_indices), rule_batch_size):
        batch_indices = selected_indices[start : start + rule_batch_size]
        states = np.repeat(initial_states[None, :, :, :], len(batch_indices), axis=0)
        states = states.reshape(len(batch_indices) * trials, height, width)
        batched_rules = np.repeat(catalog.lut_bits[batch_indices], trials, axis=0)

        states_backend = backend.asarray(states, dtype="uint8")
        rules_backend = backend.asarray(batched_rules, dtype="uint8")
        for _ in range(preprocess_steps):
            states_backend = backend.step_pairwise(states_backend, rules_backend)
        final_states = backend.to_numpy(states_backend).reshape(len(batch_indices), trials, height, width)

        for local_index, catalog_index in enumerate(batch_indices):
            local_agree, local_var = _measure_preprocess_quality(
                final_states[local_index], initial_labels, tied_mask, majority_metric
            )
            rule_bits = catalog.lut_bits[catalog_index].tolist()
            particle_velocity = _classify_isolated_velocity(rule_bits, holes=False)
            hole_velocity = _classify_isolated_velocity(rule_bits, holes=True)
            evaluation = PreprocessorEvaluation(
                legacy_index=int(catalog.ids[catalog_index]),
                stable_index=int(catalog.stable_indices[catalog_index]),
                stable_id=catalog.stable_ids[catalog_index],
                schedule_name="preprocessor_only",
                preprocess_steps=preprocess_steps,
                probability=float(probability),
                trials=trials,
                local_majority_agreement=local_agree,
                local_density_variance=local_var,
                isolated_particle_velocity=particle_velocity,
                isolated_hole_velocity=hole_velocity,
                mask=int(catalog.masks[catalog_index]),
                ones=int(catalog.lut_bits[catalog_index].sum()),
            )
            top_results.append(evaluation)

    top_results.sort(key=_preprocessor_rank_key)
    return top_results[:top_k]


def screen_preprocessors_balanced(
    catalog: BinaryCatalog,
    *,
    backend_name: str,
    width: int,
    height: int,
    probabilities: tuple[float, ...],
    trials: int,
    seed: int,
    preprocess_steps: int,
    majority_tail: int,
    majority_rule: str = "majority_moore",
    max_rules: int | None = None,
    legacy_indices: Iterable[int] | None = None,
    require_nontrivial: bool = True,
    rule_batch_size: int = 64,
    majority_metric: str = "moore",
    include_property_names: Iterable[str] | None = None,
    sample_property_limits: dict[str, int] | None = None,
    selection_seed: int = 0,
    top_k: int = 25,
) -> list[BalancedPreprocessorEvaluation]:
    if len(probabilities) < 2:
        raise ValueError("Balanced preprocessor screening requires at least two probabilities")

    selected_indices = _resolve_catalog_indices(
        catalog,
        max_rules=max_rules,
        legacy_indices=legacy_indices,
        require_nontrivial=require_nontrivial,
        include_property_names=include_property_names,
        sample_property_limits=sample_property_limits,
        selection_seed=selection_seed,
    )
    if not selected_indices:
        return []

    backend = create_backend(backend_name)
    if majority_rule == "majority_moore":
        majority_bits = np.asarray(moore_majority_rule_bits(), dtype=np.uint8)
    elif majority_rule == "majority_vn":
        majority_bits = np.asarray(von_neumann_majority_rule_bits(), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported majority rule {majority_rule!r}")

    scenario_initial_labels: list[np.ndarray] = []
    scenario_tied_masks: list[np.ndarray] = []
    scenario_initial_states: list[np.ndarray] = []
    for offset, probability in enumerate(probabilities):
        rng = np.random.default_rng(seed + offset)
        initial_states = (rng.random((trials, height, width)) < float(probability)).astype(np.uint8)
        trial_majorities = initial_states.reshape(trials, -1).sum(axis=1)
        total_sites = width * height
        initial_labels = np.where(trial_majorities > (total_sites / 2), 1, 0)
        tied_mask = trial_majorities * 2 == total_sites
        scenario_initial_states.append(initial_states)
        scenario_initial_labels.append(initial_labels)
        scenario_tied_masks.append(tied_mask)

    top_results: list[BalancedPreprocessorEvaluation] = []
    for start in range(0, len(selected_indices), rule_batch_size):
        batch_indices = selected_indices[start : start + rule_batch_size]
        batch_records: list[list[dict[str, float]]] = [[] for _ in batch_indices]

        for scenario_index, probability in enumerate(probabilities):
            initial_states = scenario_initial_states[scenario_index]
            initial_labels = scenario_initial_labels[scenario_index]
            tied_mask = scenario_tied_masks[scenario_index]

            states = np.repeat(initial_states[None, :, :, :], len(batch_indices), axis=0)
            states = states.reshape(len(batch_indices) * trials, height, width)
            batched_rules = np.repeat(catalog.lut_bits[batch_indices], trials, axis=0)
            majority_rules = np.repeat(majority_bits[None, :], len(batch_indices) * trials, axis=0)

            states_backend = backend.asarray(states, dtype="uint8")
            rules_backend = backend.asarray(batched_rules, dtype="uint8")
            majority_backend = backend.asarray(majority_rules, dtype="uint8")

            for _ in range(preprocess_steps):
                states_backend = backend.step_pairwise(states_backend, rules_backend)
            preprocess_states = backend.to_numpy(states_backend).reshape(len(batch_indices), trials, height, width)

            for _ in range(majority_tail):
                states_backend = backend.step_pairwise(states_backend, majority_backend)
            final_states = backend.to_numpy(states_backend).reshape(len(batch_indices), trials, height, width)

            for local_index, _catalog_index in enumerate(batch_indices):
                local_agree, local_var = _measure_preprocess_quality(
                    preprocess_states[local_index], initial_labels, tied_mask, majority_metric
                )
                (
                    final_majority_accuracy,
                    final_consensus_accuracy,
                    final_consensus_rate,
                    mean_final_density,
                ) = _measure_final_classification(final_states[local_index], initial_labels, tied_mask)
                batch_records[local_index].append(
                    {
                        "probability": float(probability),
                        "local_majority_agreement": local_agree,
                        "local_density_variance": local_var,
                        "final_majority_accuracy": final_majority_accuracy,
                        "final_consensus_accuracy": final_consensus_accuracy,
                        "final_consensus_rate": final_consensus_rate,
                        "mean_final_density": mean_final_density,
                    }
                )

        for local_index, catalog_index in enumerate(batch_indices):
            per_probability = tuple(batch_records[local_index])
            local_agreements = [item["local_majority_agreement"] for item in per_probability]
            local_variances = [item["local_density_variance"] for item in per_probability]
            final_accuracies = [item["final_majority_accuracy"] for item in per_probability]
            consensus_rates = [item["final_consensus_rate"] for item in per_probability]
            rule_bits = catalog.lut_bits[catalog_index].tolist()
            top_results.append(
                BalancedPreprocessorEvaluation(
                    stable_index=int(catalog.stable_indices[catalog_index]),
                    stable_id=catalog.stable_ids[catalog_index],
                    preprocess_steps=preprocess_steps,
                    final_steps=majority_tail,
                    probabilities=tuple(float(value) for value in probabilities),
                    trials=trials,
                    min_local_majority_agreement=float(min(local_agreements)),
                    max_local_density_variance=float(max(local_variances)),
                    min_final_majority_accuracy=float(min(final_accuracies)),
                    mean_final_majority_accuracy=float(sum(final_accuracies) / len(final_accuracies)),
                    final_majority_accuracy_gap=float(max(final_accuracies) - min(final_accuracies)),
                    min_final_consensus_rate=float(min(consensus_rates)),
                    isolated_particle_velocity=_classify_isolated_velocity(rule_bits, holes=False),
                    isolated_hole_velocity=_classify_isolated_velocity(rule_bits, holes=True),
                    mask=int(catalog.masks[catalog_index]),
                    ones=int(catalog.lut_bits[catalog_index].sum()),
                    per_probability=per_probability,
                )
            )

    top_results.sort(key=_balanced_preprocessor_rank_key)
    return top_results[:top_k]


def screen_repeated_block_preprocessors(
    catalog: BinaryCatalog,
    *,
    backend_name: str,
    width: int,
    height: int,
    probabilities: tuple[float, ...],
    trials: int,
    seed: int,
    preprocess_steps: int,
    amplifier_steps: int,
    repetitions: int,
    amplifier_name: str,
    amplifier_bits: list[int],
    max_rules: int | None = None,
    legacy_indices: Iterable[int] | None = None,
    require_nontrivial: bool = True,
    rule_batch_size: int = 64,
    majority_metric: str = "moore",
    include_property_names: Iterable[str] | None = None,
    sample_property_limits: dict[str, int] | None = None,
    selection_seed: int = 0,
    top_k: int = 25,
) -> list[RepeatedBlockEvaluation]:
    if not probabilities:
        raise ValueError("At least one probability is required")

    selected_indices = _resolve_catalog_indices(
        catalog,
        max_rules=max_rules,
        legacy_indices=legacy_indices,
        require_nontrivial=require_nontrivial,
        include_property_names=include_property_names,
        sample_property_limits=sample_property_limits,
        selection_seed=selection_seed,
    )
    if not selected_indices:
        return []

    backend = create_backend(backend_name)
    amplifier_array = np.asarray(amplifier_bits, dtype=np.uint8)
    scenario_initial_labels: list[np.ndarray] = []
    scenario_tied_masks: list[np.ndarray] = []
    scenario_initial_states: list[np.ndarray] = []
    for offset, probability in enumerate(probabilities):
        rng = np.random.default_rng(seed + offset)
        initial_states = (rng.random((trials, height, width)) < float(probability)).astype(np.uint8)
        trial_majorities = initial_states.reshape(trials, -1).sum(axis=1)
        total_sites = width * height
        initial_labels = np.where(trial_majorities > (total_sites / 2), 1, 0)
        tied_mask = trial_majorities * 2 == total_sites
        scenario_initial_states.append(initial_states)
        scenario_initial_labels.append(initial_labels)
        scenario_tied_masks.append(tied_mask)

    top_results: list[RepeatedBlockEvaluation] = []
    for start in range(0, len(selected_indices), rule_batch_size):
        batch_indices = selected_indices[start : start + rule_batch_size]
        batch_records: list[list[dict[str, float]]] = [[] for _ in batch_indices]

        for scenario_index, probability in enumerate(probabilities):
            initial_states = scenario_initial_states[scenario_index]
            initial_labels = scenario_initial_labels[scenario_index]
            tied_mask = scenario_tied_masks[scenario_index]

            states = np.repeat(initial_states[None, :, :, :], len(batch_indices), axis=0)
            states = states.reshape(len(batch_indices) * trials, height, width)
            pre_rules = np.repeat(catalog.lut_bits[batch_indices], trials, axis=0)
            amp_rules = np.repeat(amplifier_array[None, :], len(batch_indices) * trials, axis=0)

            states_backend = backend.asarray(states, dtype="uint8")
            pre_backend = backend.asarray(pre_rules, dtype="uint8")
            amp_backend = backend.asarray(amp_rules, dtype="uint8")

            preprocess_snapshot = None
            for _ in range(repetitions):
                for _ in range(preprocess_steps):
                    states_backend = backend.step_pairwise(states_backend, pre_backend)
                preprocess_snapshot = backend.to_numpy(states_backend).reshape(
                    len(batch_indices), trials, height, width
                )
                for _ in range(amplifier_steps):
                    states_backend = backend.step_pairwise(states_backend, amp_backend)
            final_states = backend.to_numpy(states_backend).reshape(len(batch_indices), trials, height, width)
            assert preprocess_snapshot is not None

            for local_index, _catalog_index in enumerate(batch_indices):
                local_agree, local_var = _measure_preprocess_quality(
                    preprocess_snapshot[local_index], initial_labels, tied_mask, majority_metric
                )
                (
                    final_majority_accuracy,
                    final_consensus_accuracy,
                    final_consensus_rate,
                    mean_final_density,
                ) = _measure_final_classification(final_states[local_index], initial_labels, tied_mask)
                batch_records[local_index].append(
                    {
                        "probability": float(probability),
                        "local_majority_agreement": local_agree,
                        "local_density_variance": local_var,
                        "final_majority_accuracy": final_majority_accuracy,
                        "final_consensus_accuracy": final_consensus_accuracy,
                        "final_consensus_rate": final_consensus_rate,
                        "mean_final_density": mean_final_density,
                    }
                )

        for local_index, catalog_index in enumerate(batch_indices):
            per_probability = tuple(batch_records[local_index])
            local_agreements = [item["local_majority_agreement"] for item in per_probability]
            local_variances = [item["local_density_variance"] for item in per_probability]
            final_accuracies = [item["final_majority_accuracy"] for item in per_probability]
            consensus_accuracies = [item["final_consensus_accuracy"] for item in per_probability]
            consensus_rates = [item["final_consensus_rate"] for item in per_probability]
            rule_bits = catalog.lut_bits[catalog_index].tolist()
            top_results.append(
                RepeatedBlockEvaluation(
                    stable_index=int(catalog.stable_indices[catalog_index]),
                    stable_id=catalog.stable_ids[catalog_index],
                    preprocess_steps=preprocess_steps,
                    amplifier_steps=amplifier_steps,
                    repetitions=repetitions,
                    amplifier_name=amplifier_name,
                    probabilities=tuple(float(value) for value in probabilities),
                    trials=trials,
                    min_final_consensus_rate=float(min(consensus_rates)),
                    min_final_consensus_accuracy=float(min(consensus_accuracies)),
                    min_final_majority_accuracy=float(min(final_accuracies)),
                    mean_final_majority_accuracy=float(sum(final_accuracies) / len(final_accuracies)),
                    final_majority_accuracy_gap=float(max(final_accuracies) - min(final_accuracies)),
                    max_local_density_variance=float(max(local_variances)),
                    min_local_majority_agreement=float(min(local_agreements)),
                    isolated_particle_velocity=_classify_isolated_velocity(rule_bits, holes=False),
                    isolated_hole_velocity=_classify_isolated_velocity(rule_bits, holes=True),
                    mask=int(catalog.masks[catalog_index]),
                    ones=int(catalog.lut_bits[catalog_index].sum()),
                    per_probability=per_probability,
                )
            )

    top_results.sort(key=_repeated_block_rank_key)
    return top_results[:top_k]


def screen_finite_switch_preprocessors(
    catalog: BinaryCatalog,
    *,
    backend_name: str,
    width: int,
    height: int,
    probabilities: tuple[float, ...],
    trials: int,
    seed: int,
    preprocess_steps_options: tuple[int, ...],
    amplifier_steps_options: tuple[int, ...],
    amplifier_name: str,
    amplifier_bits: list[int],
    max_rules: int | None = None,
    legacy_indices: Iterable[int] | None = None,
    rule_refs: Iterable[str] | None = None,
    require_nontrivial: bool = True,
    rule_batch_size: int = 64,
    majority_metric: str = "moore",
    include_property_names: Iterable[str] | None = None,
    sample_property_limits: dict[str, int] | None = None,
    selection_seed: int = 0,
    top_k: int = 25,
) -> list[FiniteSwitchEvaluation]:
    if len(probabilities) < 2:
        raise ValueError("Finite-switch screening requires at least two probabilities")
    preprocess_steps_options = tuple(sorted({int(value) for value in preprocess_steps_options if int(value) > 0}))
    amplifier_steps_options = tuple(sorted({int(value) for value in amplifier_steps_options if int(value) > 0}))
    if not preprocess_steps_options:
        raise ValueError("At least one preprocess step count is required")
    if not amplifier_steps_options:
        raise ValueError("At least one amplifier step count is required")

    selected_indices = _resolve_catalog_indices(
        catalog,
        max_rules=max_rules,
        legacy_indices=legacy_indices,
        rule_refs=rule_refs,
        require_nontrivial=require_nontrivial,
        include_property_names=include_property_names,
        sample_property_limits=sample_property_limits,
        selection_seed=selection_seed,
    )
    if not selected_indices:
        return []

    backend = create_backend(backend_name)
    amplifier_array = np.asarray(amplifier_bits, dtype=np.uint8)
    max_preprocess_steps = max(preprocess_steps_options)
    max_amplifier_steps = max(amplifier_steps_options)

    scenario_initial_labels: list[np.ndarray] = []
    scenario_tied_masks: list[np.ndarray] = []
    scenario_initial_states: list[np.ndarray] = []
    for offset, probability in enumerate(probabilities):
        rng = np.random.default_rng(seed + offset)
        initial_states = (rng.random((trials, height, width)) < float(probability)).astype(np.uint8)
        trial_majorities = initial_states.reshape(trials, -1).sum(axis=1)
        total_sites = width * height
        initial_labels = np.where(trial_majorities > (total_sites / 2), 1, 0)
        tied_mask = trial_majorities * 2 == total_sites
        scenario_initial_states.append(initial_states)
        scenario_initial_labels.append(initial_labels)
        scenario_tied_masks.append(tied_mask)

    top_results: list[FiniteSwitchEvaluation] = []
    for start in range(0, len(selected_indices), rule_batch_size):
        batch_indices = selected_indices[start : start + rule_batch_size]
        batch_records: dict[tuple[int, int, int], list[dict[str, float]]] = {}
        for local_index in range(len(batch_indices)):
            for preprocess_steps in preprocess_steps_options:
                for amplifier_steps in amplifier_steps_options:
                    batch_records[(local_index, preprocess_steps, amplifier_steps)] = []

        for scenario_index, probability in enumerate(probabilities):
            initial_states = scenario_initial_states[scenario_index]
            initial_labels = scenario_initial_labels[scenario_index]
            tied_mask = scenario_tied_masks[scenario_index]

            states = np.repeat(initial_states[None, :, :, :], len(batch_indices), axis=0)
            states = states.reshape(len(batch_indices) * trials, height, width)
            pre_rules = np.repeat(catalog.lut_bits[batch_indices], trials, axis=0)
            amp_rules = np.repeat(amplifier_array[None, :], len(batch_indices) * trials, axis=0)

            states_backend = backend.asarray(states, dtype="uint8")
            pre_backend = backend.asarray(pre_rules, dtype="uint8")
            amp_backend = backend.asarray(amp_rules, dtype="uint8")

            preprocess_snapshots: dict[int, np.ndarray] = {}
            for step in range(1, max_preprocess_steps + 1):
                states_backend = backend.step_pairwise(states_backend, pre_backend)
                if step in preprocess_steps_options:
                    preprocess_snapshots[step] = backend.to_numpy(states_backend).reshape(
                        len(batch_indices), trials, height, width
                    )

            for preprocess_steps in preprocess_steps_options:
                preprocess_states = preprocess_snapshots[preprocess_steps]
                amp_states_backend = backend.asarray(
                    preprocess_states.reshape(len(batch_indices) * trials, height, width),
                    dtype="uint8",
                )
                amplifier_snapshots: dict[int, np.ndarray] = {}
                for amp_step in range(1, max_amplifier_steps + 1):
                    amp_states_backend = backend.step_pairwise(amp_states_backend, amp_backend)
                    if amp_step in amplifier_steps_options:
                        amplifier_snapshots[amp_step] = backend.to_numpy(amp_states_backend).reshape(
                            len(batch_indices), trials, height, width
                        )

                for amplifier_steps in amplifier_steps_options:
                    final_states = amplifier_snapshots[amplifier_steps]
                    for local_index, _catalog_index in enumerate(batch_indices):
                        local_agree, local_var = _measure_preprocess_quality(
                            preprocess_states[local_index], initial_labels, tied_mask, majority_metric
                        )
                        (
                            final_majority_accuracy,
                            final_consensus_accuracy,
                            final_consensus_rate,
                            mean_final_density,
                        ) = _measure_final_classification(final_states[local_index], initial_labels, tied_mask)
                        batch_records[(local_index, preprocess_steps, amplifier_steps)].append(
                            {
                                "probability": float(probability),
                                "local_majority_agreement": local_agree,
                                "local_density_variance": local_var,
                                "final_majority_accuracy": final_majority_accuracy,
                                "final_consensus_accuracy": final_consensus_accuracy,
                                "final_consensus_rate": final_consensus_rate,
                                "mean_final_density": mean_final_density,
                            }
                        )

        for local_index, catalog_index in enumerate(batch_indices):
            rule_bits = catalog.lut_bits[catalog_index].tolist()
            particle_velocity = _classify_isolated_velocity(rule_bits, holes=False)
            hole_velocity = _classify_isolated_velocity(rule_bits, holes=True)
            for preprocess_steps in preprocess_steps_options:
                for amplifier_steps in amplifier_steps_options:
                    per_probability = tuple(batch_records[(local_index, preprocess_steps, amplifier_steps)])
                    local_agreements = [item["local_majority_agreement"] for item in per_probability]
                    local_variances = [item["local_density_variance"] for item in per_probability]
                    final_accuracies = [item["final_majority_accuracy"] for item in per_probability]
                    consensus_accuracies = [item["final_consensus_accuracy"] for item in per_probability]
                    consensus_rates = [item["final_consensus_rate"] for item in per_probability]
                    top_results.append(
                        FiniteSwitchEvaluation(
                            stable_index=int(catalog.stable_indices[catalog_index]),
                            stable_id=catalog.stable_ids[catalog_index],
                            preprocess_steps=preprocess_steps,
                            amplifier_steps=amplifier_steps,
                            amplifier_name=amplifier_name,
                            probabilities=tuple(float(value) for value in probabilities),
                            trials=trials,
                            min_final_consensus_rate=float(min(consensus_rates)),
                            min_final_consensus_accuracy=float(min(consensus_accuracies)),
                            min_final_majority_accuracy=float(min(final_accuracies)),
                            mean_final_majority_accuracy=float(sum(final_accuracies) / len(final_accuracies)),
                            final_majority_accuracy_gap=float(max(final_accuracies) - min(final_accuracies)),
                            max_local_density_variance=float(max(local_variances)),
                            min_local_majority_agreement=float(min(local_agreements)),
                            isolated_particle_velocity=particle_velocity,
                            isolated_hole_velocity=hole_velocity,
                            mask=int(catalog.masks[catalog_index]),
                            ones=int(catalog.lut_bits[catalog_index].sum()),
                            per_probability=per_probability,
                        )
                    )

    top_results.sort(key=_finite_switch_rank_key)
    return top_results[:top_k]


def select_catalog_indices_by_property_names(
    catalog: BinaryCatalog,
    property_names: Iterable[str],
) -> list[int]:
    property_names = tuple(dict.fromkeys(property_names))
    if not property_names:
        return []
    name_to_bit = {prop.name: prop.bit for prop in catalog.properties}
    missing = sorted(set(property_names) - set(name_to_bit))
    if missing:
        raise KeyError(f"Unknown property names: {', '.join(missing)}")

    bits = tuple(name_to_bit[name] for name in property_names)
    selected = [
        index
        for index, mask in enumerate(catalog.masks.tolist())
        if any(int(mask) & (1 << bit) for bit in bits)
    ]
    return selected


def _tile_rule_for_batch(rule_bits: list[int], batch: int, backend_name: str):
    backend = create_backend(backend_name)
    base = np.asarray(rule_bits, dtype=np.uint8)
    tiled = np.broadcast_to(base, (batch, base.shape[0]))
    return backend.asarray(tiled, dtype="uint8")


def _embedded_cardinal_traffic(velocity: str) -> list[int]:
    if velocity not in {"east", "west", "north", "south"}:
        raise ValueError(f"Unsupported cardinal traffic velocity: {velocity}")
    source_lookup = {
        "east": "t",
        "west": "w",
        "north": "b",
        "south": "y",
    }
    destination_lookup = {
        "east": "w",
        "west": "t",
        "north": "y",
        "south": "b",
    }
    pos_to_shift = {
        "x": 8,
        "y": 7,
        "z": 6,
        "t": 5,
        "u": 4,
        "w": 3,
        "a": 2,
        "b": 1,
        "c": 0,
    }
    source = source_lookup[velocity]
    destination = destination_lookup[velocity]

    bits = [0] * 512
    for index in range(512):
        source_value = (index >> pos_to_shift[source]) & 1
        center = (index >> 4) & 1
        destination_value = (index >> pos_to_shift[destination]) & 1
        bits[index] = int((source_value and not center) or (center and destination_value))
    return bits


def _resolve_catalog_indices(
    catalog: BinaryCatalog,
    *,
    max_rules: int | None,
    legacy_indices: Iterable[int] | None,
    rule_refs: Iterable[str] | None = None,
    require_nontrivial: bool,
    include_property_names: Iterable[str] | None = None,
    sample_property_limits: dict[str, int] | None = None,
    selection_seed: int = 0,
) -> list[int]:
    if rule_refs is not None:
        selected = [catalog.resolve_rule_ref(value) for value in rule_refs]
    elif legacy_indices is not None:
        selected = [catalog.resolve_rule_ref(int(value)) for value in legacy_indices]
    elif include_property_names or sample_property_limits:
        selected_set: set[int] = set()
        if include_property_names:
            selected_set.update(select_catalog_indices_by_property_names(catalog, include_property_names))
        if sample_property_limits:
            rng = np.random.default_rng(selection_seed)
            for property_name, limit in sample_property_limits.items():
                matches = select_catalog_indices_by_property_names(catalog, (property_name,))
                if limit < len(matches):
                    matches = rng.choice(np.asarray(matches, dtype=np.int64), size=limit, replace=False).tolist()
                selected_set.update(int(index) for index in matches)
        selected = sorted(selected_set)
    else:
        selected = list(range(len(catalog.ids)))
        if max_rules is not None:
            selected = selected[:max_rules]

    if max_rules is not None and (legacy_indices is not None or include_property_names or sample_property_limits):
        selected = selected[:max_rules]

    if not require_nontrivial:
        return selected

    out: list[int] = []
    for index in selected:
        sid = catalog.stable_ids[index]
        if sid in TRIVIAL_STABLE_IDS:
            continue
        out.append(index)
    return out


def _classify_isolated_velocity(bits: list[int], *, holes: bool) -> str | None:
    if holes:
        candidates = [
            velocity
            for velocity, position in VELOCITY_TO_SOURCE_POSITION.items()
            if bits[_single_zero_index(position)] == 0
        ]
    else:
        candidates = [
            velocity
            for velocity, position in VELOCITY_TO_SOURCE_POSITION.items()
            if bits[_singleton_index(position)] == 1
        ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _singleton_index(position: str) -> int:
    weights = {
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
    return weights[position]


def _single_zero_index(position: str) -> int:
    return 511 - _singleton_index(position)


def _preprocessor_rank_key(item: PreprocessorEvaluation) -> tuple[float, float, int]:
    return (-item.local_majority_agreement, item.local_density_variance, item.stable_index)


def _balanced_preprocessor_rank_key(
    item: BalancedPreprocessorEvaluation,
) -> tuple[float, float, float, float, int]:
    return (
        -item.min_final_majority_accuracy,
        item.final_majority_accuracy_gap,
        -item.min_local_majority_agreement,
        item.max_local_density_variance,
        item.stable_index,
    )


def _repeated_block_rank_key(
    item: RepeatedBlockEvaluation,
) -> tuple[float, float, float, float, float, int]:
    return (
        -item.min_final_consensus_accuracy,
        -item.min_final_consensus_rate,
        -item.min_final_majority_accuracy,
        item.final_majority_accuracy_gap,
        item.max_local_density_variance,
        item.stable_index,
    )


def _finite_switch_rank_key(
    item: FiniteSwitchEvaluation,
) -> tuple[float, float, float, float, float, int]:
    return (
        -item.min_final_consensus_accuracy,
        -item.min_final_consensus_rate,
        -item.min_final_majority_accuracy,
        item.final_majority_accuracy_gap,
        item.max_local_density_variance,
        item.stable_index,
    )


def _measure_preprocess_quality(
    states: np.ndarray,
    initial_labels: np.ndarray,
    tied_mask: np.ndarray,
    majority_metric: str,
) -> tuple[float, float]:
    local_majority = _local_majority_field(states, metric=majority_metric)
    valid = ~tied_mask
    if not np.any(valid):
        return 0.0, 0.0
    agreement = (local_majority[valid] == initial_labels[valid, None, None]).mean()

    local_density = _local_density_field(states, metric=majority_metric)
    variance = local_density[valid].var(axis=(1, 2)).mean()
    return float(agreement), float(variance)


def _measure_checkerboard_structure(states: np.ndarray) -> tuple[float, float, float]:
    states = np.asarray(states, dtype=np.uint8)
    if states.ndim != 3:
        raise ValueError("states must have shape (batch, height, width)")

    _, height, width = states.shape
    parity = (np.indices((height, width)).sum(axis=0) & 1).astype(np.int8)
    checker = 1 - 2 * parity
    spins = states.astype(np.int8) * 2 - 1
    staggered = np.abs((spins * checker[None, :, :]).mean(axis=(1, 2)))

    disagree_vertical = np.not_equal(states, np.roll(states, -1, axis=1)).mean(axis=(1, 2))
    disagree_horizontal = np.not_equal(states, np.roll(states, -1, axis=2)).mean(axis=(1, 2))
    orthogonal_disagreement = 0.5 * (disagree_vertical + disagree_horizontal)

    top_left = states
    top_right = np.roll(states, -1, axis=2)
    bottom_left = np.roll(states, -1, axis=1)
    bottom_right = np.roll(np.roll(states, -1, axis=1), -1, axis=2)
    checker_2x2 = (
        (top_left == bottom_right)
        & (top_right == bottom_left)
        & (top_left != top_right)
    )
    checker_2x2_fraction = checker_2x2.mean(axis=(1, 2))

    return (
        float(staggered.mean()),
        float(orthogonal_disagreement.mean()),
        float(checker_2x2_fraction.mean()),
    )


def _measure_final_classification(
    final_states: np.ndarray,
    initial_labels: np.ndarray,
    tied_mask: np.ndarray,
) -> tuple[float, float, float, float]:
    valid = ~tied_mask
    if not np.any(valid):
        return 0.0, 0.0, 0.0, float(final_states.mean())

    total_sites = final_states.shape[1] * final_states.shape[2]
    final_counts = final_states.reshape(final_states.shape[0], -1).sum(axis=1)
    final_labels = (final_counts > (total_sites / 2)).astype(np.uint8)
    final_majority_accuracy = (final_labels[valid] == initial_labels[valid]).mean()

    consensus_mask = (final_counts == 0) | (final_counts == total_sites)
    final_consensus_rate = consensus_mask[valid].mean()
    final_consensus_accuracy = (
        (consensus_mask[valid] & (final_labels[valid] == initial_labels[valid])).mean()
    )
    return (
        float(final_majority_accuracy),
        float(final_consensus_accuracy),
        float(final_consensus_rate),
        float(final_states.mean()),
    )


def _local_majority_field(states: np.ndarray, *, metric: str) -> np.ndarray:
    if metric == "moore":
        counts = _moore_sum(states)
        return (counts >= 5).astype(np.uint8)
    if metric == "vn":
        counts = _von_neumann_sum(states)
        return (counts >= 3).astype(np.uint8)
    raise ValueError(f"Unknown majority metric: {metric}")


def _local_density_field(states: np.ndarray, *, metric: str) -> np.ndarray:
    if metric == "moore":
        return _moore_sum(states).astype(np.float32) / 9.0
    if metric == "vn":
        return _von_neumann_sum(states).astype(np.float32) / 5.0
    raise ValueError(f"Unknown majority metric: {metric}")


def _moore_sum(states: np.ndarray) -> np.ndarray:
    return (
        np.roll(np.roll(states, 1, axis=1), 1, axis=2)
        + np.roll(states, 1, axis=1)
        + np.roll(np.roll(states, 1, axis=1), -1, axis=2)
        + np.roll(states, 1, axis=2)
        + states
        + np.roll(states, -1, axis=2)
        + np.roll(np.roll(states, -1, axis=1), 1, axis=2)
        + np.roll(states, -1, axis=1)
        + np.roll(np.roll(states, -1, axis=1), -1, axis=2)
    )


def _von_neumann_sum(states: np.ndarray) -> np.ndarray:
    return (
        np.roll(states, 1, axis=1)
        + np.roll(states, 1, axis=2)
        + states
        + np.roll(states, -1, axis=2)
        + np.roll(states, -1, axis=1)
    )
