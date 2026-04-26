from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .lut import lut_bits_to_hex, rule_stable_id_from_hex


@dataclass(frozen=True)
class PropertyInfo:
    bit: int
    name: str
    path: str


@dataclass(frozen=True)
class BinaryCatalog:
    properties: tuple[PropertyInfo, ...]
    masks: np.ndarray
    lut_bits: np.ndarray
    ids: np.ndarray
    stable_indices: np.ndarray
    stable_ids: tuple[str, ...]
    stable_order: np.ndarray
    metadata_path: Path
    binary_path: Path

    def property_names_for_mask(self, mask: int) -> tuple[str, ...]:
        return tuple(prop.name for prop in self.properties if mask & (1 << prop.bit))

    def resolve_rule_ref(self, selector: str | int) -> int:
        if isinstance(selector, (int, np.integer)):
            idx = int(selector)
            if idx < 0 or idx >= len(self.ids):
                raise KeyError(f"Legacy rule id {idx} is out of range")
            return idx

        text = str(selector).strip()
        if text.startswith("legacy:"):
            idx = int(text.split(":", 1)[1])
            if idx < 0 or idx >= len(self.ids):
                raise KeyError(f"Legacy rule id {idx} is out of range")
            return idx
        if text.startswith("rank:"):
            rank = int(text.split(":", 1)[1])
            matches = np.nonzero(self.stable_indices == rank)[0]
            if len(matches) != 1:
                raise KeyError(f"Stable rank {rank} is not unique")
            return int(matches[0])
        if text.startswith("stable:") or text.startswith("sid:"):
            prefix = text.split(":", 1)[1].lower()
            matches = [i for i, sid in enumerate(self.stable_ids) if sid.startswith(prefix)]
            if not matches:
                raise KeyError(f"No rule matches stable id prefix {prefix!r}")
            if len(matches) > 1:
                raise KeyError(f"Stable id prefix {prefix!r} is ambiguous")
            return matches[0]
        if text.isdigit():
            idx = int(text)
            if idx < 0 or idx >= len(self.ids):
                raise KeyError(f"Legacy rule id {idx} is out of range")
            return idx
        raise KeyError(
            "Unsupported rule selector. Use legacy:<n>, rank:<n>, stable:<hex-prefix>, or a bare legacy integer."
        )


def _decode_rule_bits(buf: bytes, nvars: int) -> np.ndarray:
    packed = np.frombuffer(buf, dtype=np.uint8)
    unpacked = np.unpackbits(packed, bitorder="little")
    rule_bits = unpacked[:nvars].astype(np.uint8, copy=False)
    return np.concatenate(
        (np.array([0], dtype=np.uint8), rule_bits, np.array([1], dtype=np.uint8)),
        axis=0,
    )


def load_binary_catalog(binary_path: str | Path, metadata_path: str | Path) -> BinaryCatalog:
    binary_path = Path(binary_path)
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text())

    with binary_path.open("rb") as handle:
        magic = handle.read(4)
        if magic != b"CPM1":
            raise ValueError(f"Unexpected magic {magic!r} in {binary_path}")
        header = handle.read(28)
        version, nvars, nprops, mask_bytes, solution_bytes, total_records = struct.unpack(
            "<IIIIIQ", header
        )
        if version != 1:
            raise ValueError(f"Unsupported binary version {version}")

        masks = np.empty(total_records, dtype=np.uint64)
        lut_bits = np.empty((total_records, 512), dtype=np.uint8)
        ids = np.arange(total_records, dtype=np.int64)

        for index in range(total_records):
            mask_buf = handle.read(mask_bytes)
            sol_buf = handle.read(solution_bytes)
            if len(mask_buf) != mask_bytes or len(sol_buf) != solution_bytes:
                raise ValueError(f"Truncated record {index} in {binary_path}")
            masks[index] = int.from_bytes(mask_buf, "little")
            lut_bits[index] = _decode_rule_bits(sol_buf, nvars)

    properties = tuple(
        PropertyInfo(bit=item["bit"], name=item["name"], path=item["path"])
        for item in metadata["additional_properties"]
    )

    sort_keys = [lut_bits_to_hex(bits_row.tolist()) for bits_row in lut_bits]
    stable_ids = tuple(rule_stable_id_from_hex(key) for key in sort_keys)
    stable_order = np.array(sorted(range(total_records), key=lambda i: sort_keys[i]), dtype=np.int64)
    stable_indices = np.empty(total_records, dtype=np.int64)
    for rank, idx in enumerate(stable_order.tolist()):
        stable_indices[idx] = rank

    return BinaryCatalog(
        properties=properties,
        masks=masks,
        lut_bits=lut_bits,
        ids=ids,
        stable_indices=stable_indices,
        stable_ids=stable_ids,
        stable_order=stable_order,
        metadata_path=metadata_path,
        binary_path=binary_path,
    )
