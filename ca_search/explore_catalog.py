from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .binary_catalog import load_binary_catalog
from .simple_filters import summarize_simple_rule
from .simulator import MLXBackend, NumpyBackend, _binary_entropy


DEFAULT_BINARY = Path("expanded_property_panel_nonzero.bin")
DEFAULT_METADATA = Path("expanded_property_panel_nonzero.json")


@dataclass(frozen=True)
class RuleScore:
    id: int
    mask: int
    properties: tuple[str, ...]
    tags: tuple[str, ...]
    rigid_velocity: str | None
    isolated_particle_velocity: str | None
    isolated_hole_velocity: str | None
    score: float
    mean_activity_random: float
    mean_activity_sparse: float
    final_entropy_random: float
    final_entropy_sparse: float


def _make_initial_states(batch: int, height: int, width: int, probability: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((batch, height, width)) < probability).astype(np.uint8)


def _late_window(values: np.ndarray) -> np.ndarray:
    start = values.shape[0] // 2
    return values[start:]


def _per_rule_rollout_metrics(
    backend_name: str,
    lut_bits: np.ndarray,
    initial_states: np.ndarray,
    steps: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if backend_name == "mlx":
        backend = MLXBackend()
        mx = backend.mx
    else:
        backend = NumpyBackend()
        mx = None

    n_rules = lut_bits.shape[0]
    mean_activity = np.empty(n_rules, dtype=np.float32)
    final_entropy = np.empty(n_rules, dtype=np.float32)

    for start in range(0, n_rules, batch_size):
        end = min(start + batch_size, n_rules)
        state_batch = initial_states[start:end]
        rule_batch = lut_bits[start:end]

        if backend_name == "mlx":
            states = backend.asarray(state_batch, dtype="uint8")
            rules = backend.asarray(rule_batch, dtype="uint8")
            activity_steps = []
            density_steps = []
            previous = states
            for _ in range(steps):
                states = backend.step_pairwise(states, rules)
                activity = mx.mean(mx.not_equal(states, previous), axis=(1, 2))
                density = mx.mean(states.astype(mx.float32), axis=(1, 2))
                activity_steps.append(np.asarray(activity))
                density_steps.append(np.asarray(density))
                previous = states
            activity_series = np.stack(activity_steps, axis=0)
            density_series = np.stack(density_steps, axis=0)
        else:
            states = backend.asarray(state_batch, dtype="uint8")
            rules = backend.asarray(rule_batch, dtype="uint8")
            activity_steps = []
            density_steps = []
            previous = states
            for _ in range(steps):
                states = backend.step_pairwise(states, rules)
                activity_steps.append(np.not_equal(states, previous).mean(axis=(1, 2)))
                density_steps.append(states.mean(axis=(1, 2)))
                previous = states
            activity_series = np.stack(activity_steps, axis=0)
            density_series = np.stack(density_steps, axis=0)

        mean_activity[start:end] = _late_window(activity_series).mean(axis=0)
        final_entropy[start:end] = np.array([_binary_entropy(float(p)) for p in density_series[-1]], dtype=np.float32)

    return mean_activity, final_entropy


def _simple_rule_mask(catalog) -> np.ndarray:
    keep = np.ones(catalog.ids.shape[0], dtype=bool)
    for i, bits in enumerate(catalog.lut_bits):
        summary = summarize_simple_rule(bits.tolist(), catalog.property_names_for_mask(int(catalog.masks[i])))
        if any(tag == "identity" or tag.startswith("rigid_shift:") or tag.startswith("embedded_") for tag in summary.tags):
            keep[i] = False
    return keep


def run_exploration(
    binary_path: Path,
    metadata_path: Path,
    backend: str,
    width: int,
    height: int,
    steps: int,
    batch_size: int,
    top_k: int,
) -> dict:
    catalog = load_binary_catalog(binary_path, metadata_path)
    simple_keep = _simple_rule_mask(catalog)

    random_states = _make_initial_states(catalog.ids.shape[0], height, width, 0.5, seed=0)
    sparse_states = _make_initial_states(catalog.ids.shape[0], height, width, 0.12, seed=1)

    mean_activity_random, final_entropy_random = _per_rule_rollout_metrics(
        backend, catalog.lut_bits, random_states, steps, batch_size
    )
    mean_activity_sparse, final_entropy_sparse = _per_rule_rollout_metrics(
        backend, catalog.lut_bits, sparse_states, steps, batch_size
    )

    combined_score = (
        0.4 * mean_activity_random
        + 0.3 * mean_activity_sparse
        + 0.15 * final_entropy_random
        + 0.15 * final_entropy_sparse
    )
    combined_score = np.where(simple_keep, combined_score, -1.0)

    order = np.argsort(combined_score)[::-1]
    top: list[RuleScore] = []
    for index in order[:top_k]:
        summary = summarize_simple_rule(
            catalog.lut_bits[index].tolist(),
            catalog.property_names_for_mask(int(catalog.masks[index])),
        )
        top.append(
            RuleScore(
                id=int(catalog.ids[index]),
                mask=int(catalog.masks[index]),
                properties=catalog.property_names_for_mask(int(catalog.masks[index])),
                tags=summary.tags,
                rigid_velocity=summary.rigid_velocity,
                isolated_particle_velocity=summary.isolated_particle_velocity,
                isolated_hole_velocity=summary.isolated_hole_velocity,
                score=float(combined_score[index]),
                mean_activity_random=float(mean_activity_random[index]),
                mean_activity_sparse=float(mean_activity_sparse[index]),
                final_entropy_random=float(final_entropy_random[index]),
                final_entropy_sparse=float(final_entropy_sparse[index]),
            )
        )

    return {
        "binary": str(binary_path),
        "metadata": str(metadata_path),
        "rule_count": int(catalog.ids.shape[0]),
        "nonsimple_count": int(simple_keep.sum()),
        "backend": backend,
        "grid": {"width": width, "height": height},
        "steps": steps,
        "batch_size": batch_size,
        "top_candidates": [asdict(item) for item in top],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore the expanded nonzero-mask rule catalog.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--backend", choices=("numpy", "mlx"), default="numpy")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    result = run_exploration(
        binary_path=args.binary,
        metadata_path=args.metadata,
        backend=args.backend,
        width=args.width,
        height=args.height,
        steps=args.steps,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    payload = json.dumps(result, indent=2)
    if args.out is None:
        print(payload)
    else:
        args.out.write_text(payload)
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
