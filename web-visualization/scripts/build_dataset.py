#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ca_search.binary_catalog import load_binary_catalog
from ca_search.lut import lut_bits_to_hex


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "expanded_property_panel_nonzero.bin"
DEFAULT_METADATA = REPO_ROOT / "expanded_property_panel_nonzero.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "web-visualization" / "data"
DEFAULT_INDEX_NAME = "rules-index.json"
DEFAULT_DETAIL_DIR_NAME = "rule-details"
DEFAULT_SHARD_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the browser-facing CA rule catalog as a lightweight index plus detail shards."
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--detail-dir-name", default=DEFAULT_DETAIL_DIR_NAME)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be positive")

    catalog = load_binary_catalog(args.binary, args.metadata)
    metadata = json.loads(Path(args.metadata).read_text())

    output_dir = args.output_dir
    detail_dir = output_dir / args.detail_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)

    for old_shard in detail_dir.glob("shard-*.json"):
        old_shard.unlink()

    order = catalog.stable_order.tolist()
    rules_index: list[list[object]] = []
    shard_records: list[dict[str, object]] = []
    shard_id = 0

    for stable_index, catalog_index in enumerate(order):
        mask = int(catalog.masks[catalog_index])
        lut_hex = lut_bits_to_hex(catalog.lut_bits[catalog_index].tolist())
        stable_id = catalog.stable_ids[catalog_index]

        rules_index.append(
            [
                int(catalog.ids[catalog_index]),
                stable_index,
                stable_id,
                mask,
                int(catalog.lut_bits[catalog_index].sum()),
                shard_id,
            ]
        )
        shard_records.append(
            {
                "stableId": stable_id,
                "lutHex": lut_hex,
            }
        )

        if len(shard_records) >= args.shard_size:
            write_shard(detail_dir, shard_id, shard_records)
            shard_records = []
            shard_id += 1

    if shard_records:
        write_shard(detail_dir, shard_id, shard_records)
        shard_id += 1

    properties = [
        {
            "bit": int(item["bit"]),
            "name": item["name"],
            "path": item["path"],
        }
        for item in metadata["additional_properties"]
    ]
    property_counts = [
        {
            "bit": int(item["bit"]),
            "name": item["name"],
            "count": int(item["count"]),
        }
        for item in metadata["property_counts"]
    ]
    exact_mask_counts = [
        {
            "mask": int(item["mask"]),
            "maskHex": item["mask_hex"],
            "count": int(item["count"]),
        }
        for item in metadata["exact_mask_counts"]
    ]

    index_payload = {
        "format": "ca-rule-visualization-v2",
        "summary": {
            "baseProperty": metadata.get("base_property", "simplified_equations.txt"),
            "totalBaseSolutions": int(metadata["total_base_solutions"]),
            "nonzeroMaskRuleCount": int(metadata["written_solution_records"]),
            "lutBits": 512,
            "maskBits": len(properties),
        },
        "properties": properties,
        "propertyCounts": property_counts,
        "exactMaskCounts": exact_mask_counts,
        "ruleColumns": ["legacyIndex", "stableIndex", "stableId", "mask", "ones", "detailShard"],
        "detailShards": {
            "dir": args.detail_dir_name,
            "count": shard_id,
            "size": args.shard_size,
            "pattern": f"{args.detail_dir_name}/shard-{{shard}}.json",
        },
        "rules": rules_index,
    }
    (output_dir / args.index_name).write_text(json.dumps(index_payload, separators=(",", ":")))


def write_shard(detail_dir: Path, shard_id: int, rules: list[dict[str, object]]) -> None:
    payload = {
        "format": "ca-rule-detail-shard-v1",
        "shard": shard_id,
        "rules": rules,
    }
    shard_name = f"shard-{shard_id:04d}.json"
    (detail_dir / shard_name).write_text(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
