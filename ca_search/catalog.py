from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PropertySpec:
    bit: int
    name: str
    path: str


@dataclass(frozen=True)
class RuleRecord:
    id: int
    legacy_index: int
    stable_index: int
    stable_id: str
    stable_id_short: str
    mask: int
    lut_hex: str
    ones: int


@dataclass(frozen=True)
class RuleCatalog:
    properties: tuple[PropertySpec, ...]
    rules: tuple[RuleRecord, ...]
    source_path: Path

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(prop.name for prop in self.properties)

    def property_names_for_mask(self, mask: int) -> tuple[str, ...]:
        return tuple(prop.name for prop in self.properties if mask & (1 << prop.bit))


def load_rules_dataset(path: str | Path) -> RuleCatalog:
    dataset_path = Path(path)
    data = json.loads(dataset_path.read_text())
    properties = tuple(
        PropertySpec(bit=item["bit"], name=item["name"], path=item["path"])
        for item in data["properties"]
    )
    rules = tuple(
        RuleRecord(
            id=item["id"],
            legacy_index=item.get("legacyIndex", item["id"]),
            stable_index=item.get("stableIndex", item["id"]),
            stable_id=item.get("stableId", ""),
            stable_id_short=item.get("stableIdShort", ""),
            mask=item["mask"],
            lut_hex=item["lutHex"],
            ones=item["ones"],
        )
        for item in data["rules"]
    )
    return RuleCatalog(properties=properties, rules=rules, source_path=dataset_path)
