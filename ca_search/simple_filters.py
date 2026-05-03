from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .catalog import RuleCatalog, RuleRecord
from .lut import (
    POSITION_ORDER,
    SOURCE_POSITION_TO_VELOCITY,
    VELOCITY_TO_SOURCE_POSITION,
    lut_hex_to_bits,
    rigid_rule_bits,
    single_zero_index,
    singleton_index,
)


@dataclass(frozen=True)
class SimpleRuleSummary:
    rigid_source_position: str | None
    rigid_velocity: str | None
    isolated_particle_velocity: str | None
    isolated_hole_velocity: str | None
    tags: tuple[str, ...]


def detect_rigid_source_position(lut_bits: Iterable[int]) -> str | None:
    bits = list(lut_bits)
    for position in POSITION_ORDER:
        if bits == rigid_rule_bits(position):
            return position
    return None


def classify_isolated_particle_velocity(lut_bits: Iterable[int]) -> str | None:
    bits = list(lut_bits)
    live_targets = [
        velocity
        for velocity, position in VELOCITY_TO_SOURCE_POSITION.items()
        if bits[singleton_index(position)] == 1
    ]
    if len(live_targets) != 1:
        return None
    return live_targets[0]


def classify_isolated_hole_velocity(lut_bits: Iterable[int]) -> str | None:
    bits = list(lut_bits)
    hole_targets = [
        velocity
        for velocity, position in VELOCITY_TO_SOURCE_POSITION.items()
        if bits[single_zero_index(position)] == 0
    ]
    if len(hole_targets) != 1:
        return None
    return hole_targets[0]


def summarize_simple_rule(lut_bits: Iterable[int], property_names: Iterable[str] = ()) -> SimpleRuleSummary:
    bits = list(lut_bits)
    property_name_set = set(property_names)
    rigid_source = detect_rigid_source_position(bits)
    rigid_velocity = None if rigid_source is None else SOURCE_POSITION_TO_VELOCITY[rigid_source]
    particle_velocity = classify_isolated_particle_velocity(bits)
    hole_velocity = classify_isolated_hole_velocity(bits)

    tags: list[str] = []
    if rigid_source == "u":
        tags.append("identity")
    elif rigid_velocity is not None:
        tags.append(f"rigid_shift:{rigid_velocity}")

    if "von_neumann" in property_name_set and not tags:
        tags.append("embedded_von_neumann_nonshift")
    if ("diagonal_von_neumann" in property_name_set or "diagonal_only" in property_name_set) and not tags:
        tags.append("embedded_diagonal_von_neumann_nonshift")

    if particle_velocity is not None:
        tags.append(f"isolated_particle:{particle_velocity}")
    if hole_velocity is not None:
        tags.append(f"isolated_hole:{hole_velocity}")

    if not tags:
        tags.append("unclassified")

    return SimpleRuleSummary(
        rigid_source_position=rigid_source,
        rigid_velocity=rigid_velocity,
        isolated_particle_velocity=particle_velocity,
        isolated_hole_velocity=hole_velocity,
        tags=tuple(tags),
    )


def summarize_catalog_rule(catalog: RuleCatalog, rule: RuleRecord) -> SimpleRuleSummary:
    property_names = catalog.property_names_for_mask(rule.mask)
    return summarize_simple_rule(lut_hex_to_bits(rule.lut_hex), property_names)


def count_simple_tags(catalog: RuleCatalog) -> Counter[str]:
    counts: Counter[str] = Counter()
    for rule in catalog.rules:
        summary = summarize_catalog_rule(catalog, rule)
        for tag in summary.tags:
            counts[tag] += 1
    return counts

