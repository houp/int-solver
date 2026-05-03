from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .density_classification import ScheduleSpec, evaluate_schedule
from .density_schedule_search import make_prephase_repeated_block_schedules


@dataclass(frozen=True)
class RankedCheckerboardSchedule:
    schedule_name: str
    probabilities: tuple[float, ...]
    trials: int
    min_final_majority_accuracy: float
    mean_final_majority_accuracy: float
    final_majority_accuracy_gap: float
    min_final_consensus_rate: float
    min_final_consensus_accuracy: float
    min_checkerboard_alignment: float
    min_orthogonal_disagreement: float
    min_checkerboard_2x2_fraction: float
    per_probability: tuple[dict[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_checkerboard_block_library() -> tuple[tuple[str, ...], ...]:
    return (
        ("traffic_east", "traffic_north", "traffic_west", "traffic_south"),
        ("diag_traffic_ne", "diag_traffic_nw", "diag_traffic_sw", "diag_traffic_se"),
        ("traffic_east", "traffic_north"),
        ("traffic_east", "diag_traffic_ne", "traffic_north"),
        ("cand_029b09cea",),
        ("cand_812b7dae7",),
        ("cand_58ed6b657",),
        ("cand_6774f7e43",),
        ("cand_3fd35ca10",),
        ("cand_b9b31aebc",),
        ("traffic_east", "traffic_north", "cand_029b09cea"),
        ("traffic_east", "traffic_north", "cand_812b7dae7"),
        ("diag_traffic_ne", "diag_traffic_nw", "cand_6774f7e43"),
        ("diag_traffic_ne", "diag_traffic_nw", "cand_b9b31aebc"),
    )


def build_checkerboard_prephase_schedules(
    *,
    blocks: Iterable[tuple[str, ...]],
    steps_per_rule_values: Iterable[int],
    block_repetitions: Iterable[int],
    amplifier_name: str,
    amplifier_steps_values: Iterable[int],
) -> list[ScheduleSpec]:
    schedules = []
    for steps_per_rule in steps_per_rule_values:
        for amplifier_steps in amplifier_steps_values:
            schedules.extend(
                make_prephase_repeated_block_schedules(
                    blocks,
                    (amplifier_name,),
                    steps_per_rule=int(steps_per_rule),
                    block_repetitions=tuple(int(value) for value in block_repetitions),
                    amplifier_steps=int(amplifier_steps),
                )
            )
    return schedules


def rank_checkerboard_prephase_schedules(
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
) -> list[RankedCheckerboardSchedule]:
    if not probabilities:
        raise ValueError("At least one probability is required")

    initial_state_bank: dict[float, np.ndarray] = {}
    for offset, probability in enumerate(probabilities):
        local_rng = np.random.default_rng(seed + offset)
        initial_state_bank[probability] = (
            local_rng.random((trials_per_probability, height, width)) < float(probability)
        ).astype(np.uint8)

    ranked: list[RankedCheckerboardSchedule] = []
    for schedule in schedules:
        per_probability: list[dict[str, float]] = []
        final_majority_accuracies: list[float] = []
        consensus_rates: list[float] = []
        consensus_accuracies: list[float] = []
        checker_alignments: list[float] = []
        orthogonal_disagreements: list[float] = []
        checker_2x2: list[float] = []

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
                    "preprocess_checkerboard_alignment": report.preprocess_checkerboard_alignment,
                    "preprocess_orthogonal_disagreement": report.preprocess_orthogonal_disagreement,
                    "preprocess_checkerboard_2x2_fraction": report.preprocess_checkerboard_2x2_fraction,
                    "local_majority_agreement": report.local_majority_agreement,
                    "local_density_variance": report.local_density_variance,
                    "final_majority_accuracy": report.final_majority_accuracy,
                    "final_consensus_rate": report.final_consensus_rate,
                    "final_consensus_accuracy": report.final_consensus_accuracy,
                    "mean_final_density": report.mean_final_density,
                }
            )
            final_majority_accuracies.append(report.final_majority_accuracy)
            consensus_rates.append(report.final_consensus_rate)
            consensus_accuracies.append(report.final_consensus_accuracy)
            checker_alignments.append(report.preprocess_checkerboard_alignment)
            orthogonal_disagreements.append(report.preprocess_orthogonal_disagreement)
            checker_2x2.append(report.preprocess_checkerboard_2x2_fraction)

        ranked.append(
            RankedCheckerboardSchedule(
                schedule_name=schedule.name,
                probabilities=tuple(float(value) for value in probabilities),
                trials=trials_per_probability,
                min_final_majority_accuracy=float(min(final_majority_accuracies)),
                mean_final_majority_accuracy=float(sum(final_majority_accuracies) / len(final_majority_accuracies)),
                final_majority_accuracy_gap=float(max(final_majority_accuracies) - min(final_majority_accuracies)),
                min_final_consensus_rate=float(min(consensus_rates)),
                min_final_consensus_accuracy=float(min(consensus_accuracies)),
                min_checkerboard_alignment=float(min(checker_alignments)),
                min_orthogonal_disagreement=float(min(orthogonal_disagreements)),
                min_checkerboard_2x2_fraction=float(min(checker_2x2)),
                per_probability=tuple(per_probability),
            )
        )

    ranked.sort(key=_checkerboard_schedule_rank_key)
    return ranked[:top_k]


def _checkerboard_schedule_rank_key(
    item: RankedCheckerboardSchedule,
) -> tuple[float, float, float, float, float, float, str]:
    return (
        -item.min_final_consensus_accuracy,
        -item.min_final_majority_accuracy,
        -item.min_checkerboard_2x2_fraction,
        -item.min_orthogonal_disagreement,
        -item.min_checkerboard_alignment,
        item.final_majority_accuracy_gap,
        item.schedule_name,
    )
