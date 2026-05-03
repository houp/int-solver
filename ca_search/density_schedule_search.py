from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .density_classification import ScheduleSpec, evaluate_schedule


@dataclass(frozen=True)
class RankedSchedule:
    schedule_name: str
    probabilities: tuple[float, ...]
    trials: int
    min_final_majority_accuracy: float
    mean_final_majority_accuracy: float
    final_majority_accuracy_gap: float
    min_final_consensus_rate: float
    max_local_density_variance: float
    min_local_majority_agreement: float
    per_probability: tuple[dict[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_density_search_rule_bank(
    base_bank: dict[str, list[int]],
    *,
    include_majority: bool = True,
) -> dict[str, list[int]]:
    bank = dict(base_bank)
    return bank


def make_single_preprocess_schedules(
    preprocessors: Iterable[str],
    amplifiers: Iterable[str],
    *,
    preprocess_steps: int,
    amplifier_steps: int,
) -> list[ScheduleSpec]:
    schedules: list[ScheduleSpec] = []
    for amplifier in amplifiers:
        schedules.append(ScheduleSpec(f"{amplifier}_only_{amplifier_steps}", ((amplifier, amplifier_steps),)))
        for pre in preprocessors:
            schedules.append(
                ScheduleSpec(
                    f"{pre}_{preprocess_steps}_then_{amplifier}_{amplifier_steps}",
                    ((pre, preprocess_steps), (amplifier, amplifier_steps)),
                )
            )
    return schedules


def make_preprocessor_chain_schedules(
    preprocessors: Iterable[str],
    amplifiers: Iterable[str],
    *,
    preprocess_steps: int,
    amplifier_steps: int,
    allow_repeat: bool = False,
) -> list[ScheduleSpec]:
    names = tuple(preprocessors)
    schedules: list[ScheduleSpec] = []
    for amplifier in amplifiers:
        for left in names:
            for right in names:
                if not allow_repeat and left == right:
                    continue
                schedules.append(
                    ScheduleSpec(
                        f"{left}_{preprocess_steps}_then_{right}_{preprocess_steps}_then_{amplifier}_{amplifier_steps}",
                        ((left, preprocess_steps), (right, preprocess_steps), (amplifier, amplifier_steps)),
                    )
                )
    return schedules


def make_repeated_block_schedules(
    preprocessors: Iterable[str],
    amplifiers: Iterable[str],
    *,
    preprocess_steps: int,
    amplifier_steps: int,
    repetitions: Iterable[int],
) -> list[ScheduleSpec]:
    schedules: list[ScheduleSpec] = []
    for pre in preprocessors:
        for amplifier in amplifiers:
            for reps in repetitions:
                phases: list[tuple[str, int]] = []
                for _ in range(reps):
                    phases.append((pre, preprocess_steps))
                    phases.append((amplifier, amplifier_steps))
                schedules.append(
                    ScheduleSpec(
                        f"{pre}_{preprocess_steps}_then_{amplifier}_{amplifier_steps}_x{reps}",
                        tuple(phases),
                    )
                )
    return schedules


def make_prephase_repeated_block_schedules(
    blocks: Iterable[tuple[str, ...]],
    amplifiers: Iterable[str],
    *,
    steps_per_rule: int,
    block_repetitions: Iterable[int],
    amplifier_steps: int,
) -> list[ScheduleSpec]:
    schedules: list[ScheduleSpec] = []
    for block in blocks:
        if not block:
            continue
        block_name = "_".join(block)
        for amplifier in amplifiers:
            for reps in block_repetitions:
                phases: list[tuple[str, int]] = []
                for _ in range(reps):
                    for rule_name in block:
                        phases.append((rule_name, steps_per_rule))
                phases.append((amplifier, amplifier_steps))
                schedules.append(
                    ScheduleSpec(
                        f"{block_name}_{steps_per_rule}_x{reps}_then_{amplifier}_{amplifier_steps}",
                        tuple(phases),
                    )
                )
    return schedules


def rank_density_schedules(
    backend_name: str,
    schedules: Iterable[ScheduleSpec],
    rule_bank: dict[str, list[int]],
    *,
    width: int,
    height: int,
    probabilities: tuple[float, ...],
    trials_per_probability: int,
    seed: int,
    majority_metric: str = "moore",
    top_k: int = 25,
) -> list[RankedSchedule]:
    if not probabilities:
        raise ValueError("At least one probability is required")

    rng = np.random.default_rng(seed)
    initial_state_bank: dict[float, np.ndarray] = {}
    for offset, probability in enumerate(probabilities):
        local_rng = np.random.default_rng(seed + offset)
        initial_state_bank[probability] = (
            local_rng.random((trials_per_probability, height, width)) < float(probability)
        ).astype(np.uint8)

    ranked: list[RankedSchedule] = []
    for schedule in schedules:
        per_probability: list[dict[str, float]] = []
        final_majority_accuracies: list[float] = []
        local_agreements: list[float] = []
        local_variances: list[float] = []
        consensus_rates: list[float] = []
        for probability in probabilities:
            report = evaluate_schedule(
                backend_name,
                schedule,
                rule_bank,
                initial_state_bank[probability].copy(),
                majority_metric=majority_metric,
            )
            per_probability.append(
                {
                    "probability": float(probability),
                    "local_majority_agreement": report.local_majority_agreement,
                    "local_density_variance": report.local_density_variance,
                    "preprocess_checkerboard_alignment": report.preprocess_checkerboard_alignment,
                    "preprocess_orthogonal_disagreement": report.preprocess_orthogonal_disagreement,
                    "preprocess_checkerboard_2x2_fraction": report.preprocess_checkerboard_2x2_fraction,
                    "final_majority_accuracy": report.final_majority_accuracy,
                    "final_consensus_rate": report.final_consensus_rate,
                    "final_consensus_accuracy": report.final_consensus_accuracy,
                    "mean_final_density": report.mean_final_density,
                }
            )
            final_majority_accuracies.append(report.final_majority_accuracy)
            local_agreements.append(report.local_majority_agreement)
            local_variances.append(report.local_density_variance)
            consensus_rates.append(report.final_consensus_rate)

        ranked.append(
            RankedSchedule(
                schedule_name=schedule.name,
                probabilities=tuple(float(value) for value in probabilities),
                trials=trials_per_probability,
                min_final_majority_accuracy=float(min(final_majority_accuracies)),
                mean_final_majority_accuracy=float(sum(final_majority_accuracies) / len(final_majority_accuracies)),
                final_majority_accuracy_gap=float(max(final_majority_accuracies) - min(final_majority_accuracies)),
                min_final_consensus_rate=float(min(consensus_rates)),
                max_local_density_variance=float(max(local_variances)),
                min_local_majority_agreement=float(min(local_agreements)),
                per_probability=tuple(per_probability),
            )
        )

    ranked.sort(key=_ranked_schedule_key)
    return ranked[:top_k]


def _ranked_schedule_key(item: RankedSchedule) -> tuple[float, float, float, float, str]:
    return (
        -item.min_final_consensus_rate,
        -item.min_final_majority_accuracy,
        item.final_majority_accuracy_gap,
        item.max_local_density_variance,
        item.schedule_name,
    )
