from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .binary_catalog import BinaryCatalog
from .binary_catalog import load_binary_catalog
from .catalog import load_rules_dataset
from .density_checkerboard_search import (
    build_checkerboard_prephase_schedules,
    default_checkerboard_block_library,
    rank_checkerboard_prephase_schedules,
)
from .density_classification import (
    default_density_rule_bank,
    default_density_schedules,
    evaluate_schedule_suite,
    moore_threshold_rule_bits,
    screen_finite_switch_preprocessors,
    screen_repeated_block_preprocessors,
    screen_preprocessors_balanced,
    screen_preprocessors,
    von_neumann_threshold_rule_bits,
)
from .lut import lut_hex_to_bits
from .raw_reversibility_screen import (
    DEFAULT_STAGE_POPULATIONS,
    screen_raw_catalog_for_reversibility,
    write_raw_screen_summary,
)
from .reversibility import run_exact_reversibility_screen
from .reversibility_catalog import screen_catalog_exact_reversibility
from .simple_filters import count_simple_tags, summarize_catalog_rule
from .simulator import benchmark_pairwise, collect_metric_series


DEFAULT_DATASET = Path("web-visualization/data/rules-dataset.json")
DEFAULT_BINARY = Path("expanded_property_panel_nonzero.bin")
DEFAULT_METADATA = Path("expanded_property_panel_nonzero.json")
DEFAULT_RAW_SOLUTIONS = Path("witek")


def _load_rules_from_dataset(dataset_path: Path, limit: int) -> list[list[int]]:
    catalog = load_rules_dataset(dataset_path)
    rules = [lut_hex_to_bits(rule.lut_hex) for rule in catalog.rules[:limit]]
    if not rules:
        raise RuntimeError(f"No rules found in {dataset_path}")
    return rules


def _make_random_states(batch: int, width: int, height: int, probability: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    return (rng.random((batch, height, width)) < probability).astype(np.uint8)


def _slice_binary_catalog(catalog: BinaryCatalog, limit: int) -> BinaryCatalog:
    if limit <= 0:
        raise ValueError("--max-rules must be positive")
    if limit >= len(catalog.ids):
        return catalog
    return BinaryCatalog(
        properties=catalog.properties,
        masks=catalog.masks[:limit].copy(),
        lut_bits=catalog.lut_bits[:limit].copy(),
        ids=catalog.ids[:limit].copy(),
        stable_indices=catalog.stable_indices[:limit].copy(),
        stable_ids=tuple(catalog.stable_ids[:limit]),
        stable_order=catalog.stable_order[:limit].copy(),
        metadata_path=catalog.metadata_path,
        binary_path=catalog.binary_path,
    )


def _normalize_torus_args(torus: list[list[int]] | None, fallback: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not torus:
        return tuple(fallback)
    return tuple((int(width), int(height)) for width, height in torus)


def cmd_screen_simple(args: argparse.Namespace) -> int:
    catalog = load_rules_dataset(args.dataset)
    counts = count_simple_tags(catalog)
    print(f"Dataset: {catalog.source_path}")
    print(f"Rules: {len(catalog.rules)}")
    for tag, count in sorted(counts.items()):
        print(f"{tag}: {count}")
    if args.show_first:
        print()
        for rule in catalog.rules[: args.show_first]:
            summary = summarize_catalog_rule(catalog, rule)
            print(
                f"stableIndex={rule.stable_index} stableId={rule.stable_id_short} "
                f"legacyIndex={rule.legacy_index} tags={summary.tags}"
            )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    rules = _load_rules_from_dataset(args.dataset, args.batch)
    states = _make_random_states(len(rules), args.width, args.height, args.probability, args.seed)
    result = benchmark_pairwise(args.backend, rules, states, args.steps)
    print(json.dumps(result.__dict__, indent=2))
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    rules = _load_rules_from_dataset(args.dataset, args.batch)
    states = _make_random_states(len(rules), args.width, args.height, args.probability, args.seed)
    series = collect_metric_series(args.backend, rules, states, args.steps)
    print(
        json.dumps(
            {
                "density": series.density,
                "entropy": series.entropy,
                "activity": series.activity,
            },
            indent=2,
        )
    )
    return 0


def cmd_density_classification_study(args: argparse.Namespace) -> int:
    rule_bank = default_density_rule_bank(str(args.binary), str(args.metadata))
    schedules = default_density_schedules(
        preprocess_short=args.preprocess_short,
        preprocess_long=args.preprocess_long,
        majority_tail=args.majority_tail,
        majority_name=args.majority_rule,
    )
    probabilities = tuple(float(value) for value in args.probabilities)
    reports = evaluate_schedule_suite(
        args.backend,
        schedules,
        rule_bank,
        width=args.width,
        height=args.height,
        probabilities=probabilities,
        trials_per_probability=args.trials,
        seed=args.seed,
        majority_metric=args.majority_metric,
    )
    payload = {
        "binary": str(args.binary),
        "metadata": str(args.metadata),
        "backend": args.backend,
        "grid": [args.width, args.height],
        "probabilities": list(probabilities),
        "trials": args.trials,
        "preprocess_short": args.preprocess_short,
        "preprocess_long": args.preprocess_long,
        "majority_tail": args.majority_tail,
        "majority_rule": args.majority_rule,
        "majority_metric": args.majority_metric,
        "reports": reports,
    }
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def cmd_density_preprocessor_screen(args: argparse.Namespace) -> int:
    catalog = load_binary_catalog(args.binary, args.metadata)
    report = screen_preprocessors(
        catalog,
        backend_name=args.backend,
        width=args.width,
        height=args.height,
        probability=args.probability,
        trials=args.trials,
        seed=args.seed,
        preprocess_steps=args.preprocess_steps,
        max_rules=args.max_rules,
        legacy_indices=args.legacy_indices,
        require_nontrivial=not args.include_trivial,
        rule_batch_size=args.rule_batch_size,
        majority_metric=args.majority_metric,
        top_k=args.top_k,
    )
    payload = {
        "binary": str(args.binary),
        "metadata": str(args.metadata),
        "backend": args.backend,
        "grid": [args.width, args.height],
        "probability": args.probability,
        "trials": args.trials,
        "preprocess_steps": args.preprocess_steps,
        "majority_metric": args.majority_metric,
        "top_k": args.top_k,
        "results": [item.to_dict() for item in report],
    }
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def _parse_sample_property_args(items: list[str] | None) -> dict[str, int] | None:
    if not items:
        return None
    out: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --sample-property NAME=COUNT, got {item!r}")
        name, count_text = item.split("=", 1)
        out[name.strip()] = int(count_text)
    return out


def cmd_density_preprocessor_balanced_screen(args: argparse.Namespace) -> int:
    catalog = load_binary_catalog(args.binary, args.metadata)
    report = screen_preprocessors_balanced(
        catalog,
        backend_name=args.backend,
        width=args.width,
        height=args.height,
        probabilities=tuple(float(value) for value in args.probabilities),
        trials=args.trials,
        seed=args.seed,
        preprocess_steps=args.preprocess_steps,
        majority_tail=args.majority_tail,
        majority_rule=args.majority_rule,
        max_rules=args.max_rules,
        legacy_indices=args.legacy_indices,
        require_nontrivial=not args.include_trivial,
        rule_batch_size=args.rule_batch_size,
        majority_metric=args.majority_metric,
        include_property_names=args.include_property,
        sample_property_limits=_parse_sample_property_args(args.sample_property),
        selection_seed=args.selection_seed,
        top_k=args.top_k,
    )
    payload = {
        "binary": str(args.binary),
        "metadata": str(args.metadata),
        "backend": args.backend,
        "grid": [args.width, args.height],
        "probabilities": [float(value) for value in args.probabilities],
        "trials": args.trials,
        "preprocess_steps": args.preprocess_steps,
        "majority_tail": args.majority_tail,
        "majority_rule": args.majority_rule,
        "majority_metric": args.majority_metric,
        "include_property": list(args.include_property or []),
        "sample_property": _parse_sample_property_args(args.sample_property) or {},
        "selection_seed": args.selection_seed,
        "top_k": args.top_k,
        "results": [item.to_dict() for item in report],
    }
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def _resolve_amplifier_rule_bits(name: str) -> list[int]:
    if name == "majority_moore":
        return moore_threshold_rule_bits(5)
    if name == "majority_vn":
        return von_neumann_threshold_rule_bits(3)
    if name.startswith("moore_threshold:"):
        return moore_threshold_rule_bits(int(name.split(":", 1)[1]))
    if name.startswith("vn_threshold:"):
        return von_neumann_threshold_rule_bits(int(name.split(":", 1)[1]))
    raise ValueError(
        "Unsupported amplifier name. Use majority_moore, majority_vn, moore_threshold:<n>, or vn_threshold:<n>."
    )


def cmd_density_repeated_block_screen(args: argparse.Namespace) -> int:
    catalog = load_binary_catalog(args.binary, args.metadata)
    amplifier_bits = _resolve_amplifier_rule_bits(args.amplifier)
    report = screen_repeated_block_preprocessors(
        catalog,
        backend_name=args.backend,
        width=args.width,
        height=args.height,
        probabilities=tuple(float(value) for value in args.probabilities),
        trials=args.trials,
        seed=args.seed,
        preprocess_steps=args.preprocess_steps,
        amplifier_steps=args.amplifier_steps,
        repetitions=args.repetitions,
        amplifier_name=args.amplifier,
        amplifier_bits=amplifier_bits,
        max_rules=args.max_rules,
        legacy_indices=args.legacy_indices,
        rule_refs=args.rule_refs,
        require_nontrivial=not args.include_trivial,
        rule_batch_size=args.rule_batch_size,
        majority_metric=args.majority_metric,
        include_property_names=args.include_property,
        sample_property_limits=_parse_sample_property_args(args.sample_property),
        selection_seed=args.selection_seed,
        top_k=args.top_k,
    )
    payload = {
        "binary": str(args.binary),
        "metadata": str(args.metadata),
        "backend": args.backend,
        "grid": [args.width, args.height],
        "probabilities": [float(value) for value in args.probabilities],
        "trials": args.trials,
        "preprocess_steps": args.preprocess_steps,
        "amplifier_steps": args.amplifier_steps,
        "repetitions": args.repetitions,
        "amplifier": args.amplifier,
        "majority_metric": args.majority_metric,
        "include_property": list(args.include_property or []),
        "sample_property": _parse_sample_property_args(args.sample_property) or {},
        "selection_seed": args.selection_seed,
        "top_k": args.top_k,
        "results": [item.to_dict() for item in report],
    }
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def cmd_density_finite_switch_screen(args: argparse.Namespace) -> int:
    catalog = load_binary_catalog(args.binary, args.metadata)
    amplifier_bits = _resolve_amplifier_rule_bits(args.amplifier)
    report = screen_finite_switch_preprocessors(
        catalog,
        backend_name=args.backend,
        width=args.width,
        height=args.height,
        probabilities=tuple(float(value) for value in args.probabilities),
        trials=args.trials,
        seed=args.seed,
        preprocess_steps_options=tuple(args.preprocess_steps),
        amplifier_steps_options=tuple(args.amplifier_steps),
        amplifier_name=args.amplifier,
        amplifier_bits=amplifier_bits,
        max_rules=args.max_rules,
        legacy_indices=args.legacy_indices,
        require_nontrivial=not args.include_trivial,
        rule_batch_size=args.rule_batch_size,
        majority_metric=args.majority_metric,
        include_property_names=args.include_property,
        sample_property_limits=_parse_sample_property_args(args.sample_property),
        selection_seed=args.selection_seed,
        top_k=args.top_k,
    )
    payload = {
        "binary": str(args.binary),
        "metadata": str(args.metadata),
        "backend": args.backend,
        "grid": [args.width, args.height],
        "probabilities": [float(value) for value in args.probabilities],
        "trials": args.trials,
        "preprocess_steps": list(args.preprocess_steps),
        "amplifier_steps": list(args.amplifier_steps),
        "amplifier": args.amplifier,
        "majority_metric": args.majority_metric,
        "include_property": list(args.include_property or []),
        "sample_property": _parse_sample_property_args(args.sample_property) or {},
        "selection_seed": args.selection_seed,
        "top_k": args.top_k,
        "results": [item.to_dict() for item in report],
    }
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def cmd_density_checkerboard_prephase_screen(args: argparse.Namespace) -> int:
    rule_bank = default_density_rule_bank(str(args.binary), str(args.metadata))
    schedules = build_checkerboard_prephase_schedules(
        blocks=default_checkerboard_block_library(),
        steps_per_rule_values=tuple(args.steps_per_rule),
        block_repetitions=tuple(args.block_repetitions),
        amplifier_name=args.amplifier,
        amplifier_steps_values=tuple(args.amplifier_steps),
    )
    report = rank_checkerboard_prephase_schedules(
        args.backend,
        schedules,
        rule_bank,
        width=args.width,
        height=args.height,
        probabilities=tuple(float(value) for value in args.probabilities),
        trials_per_probability=args.trials,
        seed=args.seed,
        majority_metric=args.majority_metric,
        top_k=args.top_k,
    )
    payload = {
        "binary": str(args.binary),
        "metadata": str(args.metadata),
        "backend": args.backend,
        "grid": [args.width, args.height],
        "probabilities": [float(value) for value in args.probabilities],
        "trials": args.trials,
        "steps_per_rule": list(args.steps_per_rule),
        "block_repetitions": list(args.block_repetitions),
        "amplifier": args.amplifier,
        "amplifier_steps": list(args.amplifier_steps),
        "majority_metric": args.majority_metric,
        "top_k": args.top_k,
        "results": [item.to_dict() for item in report],
    }
    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


def cmd_reversibility_probe(args: argparse.Namespace) -> int:
    catalog = load_binary_catalog(args.binary, args.metadata)
    selector = args.rule
    index = catalog.resolve_rule_ref(selector)
    report = run_exact_reversibility_screen(
        catalog.lut_bits[index].tolist(),
        sector_grid=(args.sector_width, args.sector_height),
        particle_populations=tuple(args.particle_populations),
        hole_populations=tuple(args.hole_populations),
        torus_grids=_normalize_torus_args(args.torus, [(3, 3), (4, 4)]),
        sector_batch_size=args.sector_batch_size,
        torus_batch_size=args.torus_batch_size,
    ).to_dict()
    report.update(
        {
            "selector": selector,
            "resolved_legacy_index": int(catalog.ids[index]),
            "resolved_stable_index": int(catalog.stable_indices[index]),
            "resolved_stable_id": catalog.stable_ids[index],
        }
    )
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(report, indent=2))
    return 0


def cmd_reversibility_screen_catalog(args: argparse.Namespace) -> int:
    catalog = load_binary_catalog(args.binary, args.metadata)
    if args.max_rules is not None:
        catalog = _slice_binary_catalog(catalog, args.max_rules)

    report = screen_catalog_exact_reversibility(
        catalog,
        sector_grid=(args.sector_width, args.sector_height),
        particle_populations=tuple(args.particle_populations),
        hole_populations=tuple(args.hole_populations),
        torus_grids=_normalize_torus_args(args.torus, [(3, 3)]),
        rule_batch_size=args.rule_batch_size,
    ).to_dict()
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(report, indent=2))
    return 0


def cmd_reversibility_screen_raw(args: argparse.Namespace) -> int:
    summary = screen_raw_catalog_for_reversibility(
        args.input,
        output_path=args.survivor_out,
        records_per_chunk=args.records_per_chunk,
        stage_populations=tuple(args.stage_populations),
        max_survivor_summaries=args.max_survivor_summaries,
    )
    if args.out is not None:
        write_raw_screen_summary(args.out, summary)
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(summary.to_dict(), indent=2))
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search helpers for interesting number-conserving rules.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    screen = subparsers.add_parser("screen-simple", help="Classify simple analytically understood rules.")
    screen.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    screen.add_argument("--show-first", type=int, default=0)
    screen.set_defaults(func=cmd_screen_simple)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark the batched simulator.")
    benchmark.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    benchmark.add_argument("--backend", choices=("auto", "numpy", "mlx"), default="auto")
    benchmark.add_argument("--batch", type=int, default=128)
    benchmark.add_argument("--width", type=int, default=128)
    benchmark.add_argument("--height", type=int, default=128)
    benchmark.add_argument("--steps", type=int, default=128)
    benchmark.add_argument("--probability", type=float, default=0.35)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.set_defaults(func=cmd_benchmark)

    metrics = subparsers.add_parser("metrics", help="Run a short rollout and report cheap metrics.")
    metrics.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    metrics.add_argument("--backend", choices=("auto", "numpy", "mlx"), default="auto")
    metrics.add_argument("--batch", type=int, default=8)
    metrics.add_argument("--width", type=int, default=64)
    metrics.add_argument("--height", type=int, default=64)
    metrics.add_argument("--steps", type=int, default=32)
    metrics.add_argument("--probability", type=float, default=0.35)
    metrics.add_argument("--seed", type=int, default=0)
    metrics.set_defaults(func=cmd_metrics)

    density = subparsers.add_parser(
        "density-classification-study",
        help="Evaluate Fukś-style preprocess-plus-majority schedules on the 2D NCCA catalog.",
    )
    density.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    density.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    density.add_argument("--backend", choices=("auto", "numpy", "mlx"), default="auto")
    density.add_argument("--width", type=int, default=64)
    density.add_argument("--height", type=int, default=64)
    density.add_argument("--trials", type=int, default=32)
    density.add_argument(
        "--probabilities",
        type=float,
        nargs="*",
        default=[0.45, 0.475, 0.49, 0.51, 0.525, 0.55],
    )
    density.add_argument("--seed", type=int, default=0)
    density.add_argument("--preprocess-short", type=int, default=32)
    density.add_argument("--preprocess-long", type=int, default=64)
    density.add_argument("--majority-tail", type=int, default=32)
    density.add_argument(
        "--majority-rule",
        choices=("majority_moore", "majority_vn"),
        default="majority_moore",
    )
    density.add_argument("--majority-metric", choices=("moore", "vn"), default="moore")
    density.add_argument("--out", type=Path, default=None)
    density.set_defaults(func=cmd_density_classification_study)

    preprocessor = subparsers.add_parser(
        "density-preprocessor-screen",
        help="Rank candidate NCCAs by preprocessing quality for 2D density-classification schedules.",
    )
    preprocessor.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    preprocessor.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    preprocessor.add_argument("--backend", choices=("auto", "numpy", "mlx"), default="auto")
    preprocessor.add_argument("--width", type=int, default=64)
    preprocessor.add_argument("--height", type=int, default=64)
    preprocessor.add_argument("--probability", type=float, default=0.49)
    preprocessor.add_argument("--trials", type=int, default=32)
    preprocessor.add_argument("--seed", type=int, default=0)
    preprocessor.add_argument("--preprocess-steps", type=int, default=64)
    preprocessor.add_argument("--max-rules", type=int, default=None)
    preprocessor.add_argument("--legacy-indices", type=int, nargs="*", default=None)
    preprocessor.add_argument("--include-trivial", action="store_true")
    preprocessor.add_argument("--rule-batch-size", type=int, default=64)
    preprocessor.add_argument("--majority-metric", choices=("moore", "vn"), default="moore")
    preprocessor.add_argument("--top-k", type=int, default=25)
    preprocessor.add_argument("--out", type=Path, default=None)
    preprocessor.set_defaults(func=cmd_density_preprocessor_screen)

    preprocessor_balanced = subparsers.add_parser(
        "density-preprocessor-balanced-screen",
        help="Rank candidate NCCAs by balanced paired-density preprocess-plus-majority performance.",
    )
    preprocessor_balanced.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    preprocessor_balanced.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    preprocessor_balanced.add_argument("--backend", choices=("auto", "numpy", "mlx"), default="auto")
    preprocessor_balanced.add_argument("--width", type=int, default=32)
    preprocessor_balanced.add_argument("--height", type=int, default=32)
    preprocessor_balanced.add_argument("--probabilities", type=float, nargs="*", default=[0.49, 0.51])
    preprocessor_balanced.add_argument("--trials", type=int, default=16)
    preprocessor_balanced.add_argument("--seed", type=int, default=0)
    preprocessor_balanced.add_argument("--preprocess-steps", type=int, default=32)
    preprocessor_balanced.add_argument("--majority-tail", type=int, default=32)
    preprocessor_balanced.add_argument(
        "--majority-rule",
        choices=("majority_moore", "majority_vn"),
        default="majority_moore",
    )
    preprocessor_balanced.add_argument("--max-rules", type=int, default=None)
    preprocessor_balanced.add_argument("--legacy-indices", type=int, nargs="*", default=None)
    preprocessor_balanced.add_argument("--include-property", type=str, nargs="*", default=None)
    preprocessor_balanced.add_argument(
        "--sample-property",
        type=str,
        action="append",
        default=None,
        help="Add a sampled property subset as NAME=COUNT.",
    )
    preprocessor_balanced.add_argument("--selection-seed", type=int, default=0)
    preprocessor_balanced.add_argument("--include-trivial", action="store_true")
    preprocessor_balanced.add_argument("--rule-batch-size", type=int, default=64)
    preprocessor_balanced.add_argument("--majority-metric", choices=("moore", "vn"), default="moore")
    preprocessor_balanced.add_argument("--top-k", type=int, default=25)
    preprocessor_balanced.add_argument("--out", type=Path, default=None)
    preprocessor_balanced.set_defaults(func=cmd_density_preprocessor_balanced_screen)

    repeated_block = subparsers.add_parser(
        "density-repeated-block-screen",
        help="Rank candidate NCCAs under repeated [preprocessor, amplifier] block schedules.",
    )
    repeated_block.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    repeated_block.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    repeated_block.add_argument("--backend", choices=("auto", "numpy", "mlx"), default="auto")
    repeated_block.add_argument("--width", type=int, default=32)
    repeated_block.add_argument("--height", type=int, default=32)
    repeated_block.add_argument("--probabilities", type=float, nargs="*", default=[0.49, 0.51])
    repeated_block.add_argument("--trials", type=int, default=16)
    repeated_block.add_argument("--seed", type=int, default=0)
    repeated_block.add_argument("--preprocess-steps", type=int, default=8)
    repeated_block.add_argument("--amplifier-steps", type=int, default=8)
    repeated_block.add_argument("--repetitions", type=int, default=16)
    repeated_block.add_argument("--amplifier", type=str, default="majority_vn")
    repeated_block.add_argument("--max-rules", type=int, default=None)
    repeated_block.add_argument("--legacy-indices", type=int, nargs="*", default=None)
    repeated_block.add_argument("--include-property", type=str, nargs="*", default=None)
    repeated_block.add_argument(
        "--sample-property",
        type=str,
        action="append",
        default=None,
        help="Add a sampled property subset as NAME=COUNT.",
    )
    repeated_block.add_argument("--selection-seed", type=int, default=0)
    repeated_block.add_argument("--include-trivial", action="store_true")
    repeated_block.add_argument("--rule-batch-size", type=int, default=64)
    repeated_block.add_argument("--majority-metric", choices=("moore", "vn"), default="moore")
    repeated_block.add_argument("--top-k", type=int, default=25)
    repeated_block.add_argument("--out", type=Path, default=None)
    repeated_block.set_defaults(func=cmd_density_repeated_block_screen)

    finite_switch = subparsers.add_parser(
        "density-finite-switch-screen",
        help="Rank candidate NCCAs under a single global switch schedule F^(T1) -> G^(T2).",
    )
    finite_switch.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    finite_switch.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    finite_switch.add_argument("--backend", choices=("auto", "numpy", "mlx"), default="auto")
    finite_switch.add_argument("--width", type=int, default=32)
    finite_switch.add_argument("--height", type=int, default=32)
    finite_switch.add_argument("--probabilities", type=float, nargs="*", default=[0.49, 0.51])
    finite_switch.add_argument("--trials", type=int, default=16)
    finite_switch.add_argument("--seed", type=int, default=0)
    finite_switch.add_argument("--preprocess-steps", type=int, nargs="*", default=[16, 32, 64, 96])
    finite_switch.add_argument("--amplifier-steps", type=int, nargs="*", default=[16, 32, 64, 96])
    finite_switch.add_argument("--amplifier", type=str, default="majority_vn")
    finite_switch.add_argument("--max-rules", type=int, default=None)
    finite_switch.add_argument("--legacy-indices", type=int, nargs="*", default=None)
    finite_switch.add_argument("--rule-refs", type=str, nargs="*", default=None)
    finite_switch.add_argument("--include-property", type=str, nargs="*", default=None)
    finite_switch.add_argument(
        "--sample-property",
        type=str,
        action="append",
        default=None,
        help="Add a sampled property subset as NAME=COUNT.",
    )
    finite_switch.add_argument("--selection-seed", type=int, default=0)
    finite_switch.add_argument("--include-trivial", action="store_true")
    finite_switch.add_argument("--rule-batch-size", type=int, default=64)
    finite_switch.add_argument("--majority-metric", choices=("moore", "vn"), default="moore")
    finite_switch.add_argument("--top-k", type=int, default=25)
    finite_switch.add_argument("--out", type=Path, default=None)
    finite_switch.set_defaults(func=cmd_density_finite_switch_screen)

    checkerboard = subparsers.add_parser(
        "density-checkerboard-prephase-screen",
        help="Rank finite prephase schedules by checkerboard ordering and downstream classification quality.",
    )
    checkerboard.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    checkerboard.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    checkerboard.add_argument("--backend", choices=("auto", "numpy", "mlx"), default="auto")
    checkerboard.add_argument("--width", type=int, default=32)
    checkerboard.add_argument("--height", type=int, default=32)
    checkerboard.add_argument("--probabilities", type=float, nargs="*", default=[0.49, 0.51])
    checkerboard.add_argument("--trials", type=int, default=16)
    checkerboard.add_argument("--seed", type=int, default=0)
    checkerboard.add_argument("--steps-per-rule", type=int, nargs="*", default=[1, 2])
    checkerboard.add_argument("--block-repetitions", type=int, nargs="*", default=[16, 32, 64])
    checkerboard.add_argument("--amplifier", type=str, default="majority_vn")
    checkerboard.add_argument("--amplifier-steps", type=int, nargs="*", default=[32, 64])
    checkerboard.add_argument("--majority-metric", choices=("moore", "vn"), default="moore")
    checkerboard.add_argument("--top-k", type=int, default=25)
    checkerboard.add_argument("--out", type=Path, default=None)
    checkerboard.set_defaults(func=cmd_density_checkerboard_prephase_screen)

    reversibility = subparsers.add_parser(
        "reversibility-probe",
        help="Run exact early reversibility rejection tests on a selected rule from a binary catalog.",
    )
    reversibility.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    reversibility.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    reversibility.add_argument("--rule", type=str, required=True)
    reversibility.add_argument("--sector-width", type=int, default=8)
    reversibility.add_argument("--sector-height", type=int, default=8)
    reversibility.add_argument("--particle-populations", type=int, nargs="*", default=[2, 3, 4])
    reversibility.add_argument("--hole-populations", type=int, nargs="*", default=[1, 2, 3, 4])
    reversibility.add_argument(
        "--torus",
        type=int,
        nargs=2,
        action="append",
        default=None,
        metavar=("WIDTH", "HEIGHT"),
    )
    reversibility.add_argument("--sector-batch-size", type=int, default=8192)
    reversibility.add_argument("--torus-batch-size", type=int, default=16384)
    reversibility.add_argument("--out", type=Path, default=None)
    reversibility.set_defaults(func=cmd_reversibility_probe)

    reversibility_catalog = subparsers.add_parser(
        "reversibility-screen-catalog",
        help="Run exact early reversibility rejection tests over every rule in a binary catalog.",
    )
    reversibility_catalog.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    reversibility_catalog.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    reversibility_catalog.add_argument("--sector-width", type=int, default=4)
    reversibility_catalog.add_argument("--sector-height", type=int, default=4)
    reversibility_catalog.add_argument("--particle-populations", type=int, nargs="*", default=[2, 3])
    reversibility_catalog.add_argument("--hole-populations", type=int, nargs="*", default=[1, 2])
    reversibility_catalog.add_argument(
        "--torus",
        type=int,
        nargs=2,
        action="append",
        default=None,
        metavar=("WIDTH", "HEIGHT"),
    )
    reversibility_catalog.add_argument("--rule-batch-size", type=int, default=1024)
    reversibility_catalog.add_argument("--max-rules", type=int, default=None)
    reversibility_catalog.add_argument("--out", type=Path, default=None)
    reversibility_catalog.set_defaults(func=cmd_reversibility_screen_catalog)

    reversibility_raw = subparsers.add_parser(
        "reversibility-screen-raw",
        help="Stream the legacy raw solver outputs once and apply the exact 4x4 particle-sector screen.",
    )
    reversibility_raw.add_argument("--input", type=Path, default=DEFAULT_RAW_SOLUTIONS)
    reversibility_raw.add_argument("--stage-populations", type=int, nargs="*", default=list(DEFAULT_STAGE_POPULATIONS))
    reversibility_raw.add_argument("--records-per-chunk", type=int, default=16384)
    reversibility_raw.add_argument("--max-survivor-summaries", type=int, default=4096)
    reversibility_raw.add_argument("--survivor-out", type=Path, default=None)
    reversibility_raw.add_argument("--out", type=Path, default=None)
    reversibility_raw.set_defaults(func=cmd_reversibility_screen_raw)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
