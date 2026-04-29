"""Validate the best strict-CA mixture on the joint random + adversarial
test suite, mirroring the format used by classifier/scaled_schedule.py
(non-local baseline) for direct comparison.

Reports correct-consensus rate, median / p95 time-to-consensus, and
how many trials hit the step cap.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# --- repo-root path bootstrap ---
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
_CATALOG_DIR = _REPO_ROOT / "catalogs"
# --------------------------------

from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend
from strict_ca_classifier import resolve_rule_bits, run_strict_ca, _correct_consensus
from density_margin_sweep import (
    build_random_batch_at_density,
    build_adversarial_at_density,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--mixture", required=True,
                        help="rule1:p1,rule2:p2,...")
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 96, 128, 192, 256])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.02])
    parser.add_argument("--n-random-per-side", type=int, default=32)
    parser.add_argument("--max-steps-mult", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("strict_ca_validation.json"))
    args = parser.parse_args()

    parts = [seg.strip() for seg in args.mixture.split(",")]
    rule_names = []; probs = []
    for p in parts:
        name, prob = p.rsplit(":", 1)
        rule_names.append(name.strip())
        probs.append(float(prob))
    if abs(sum(probs) - 1.0) > 1e-6:
        raise SystemExit(f"probabilities must sum to 1, got {sum(probs)}")

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    rule_luts = [resolve_rule_bits(n, catalog=catalog) for n in rule_names]
    backend = create_backend(args.backend)

    runs = []
    for L in args.grids:
        max_steps = args.max_steps_mult * L
        for delta in args.deltas:
            density_above = 0.5 + delta
            density_below = 0.5 - delta
            n_total = L * L
            actual_above = int(round(density_above * n_total))
            actual_below = int(round(density_below * n_total))
            if actual_above <= n_total // 2 or actual_below >= n_total // 2:
                continue

            rng_build = np.random.default_rng(args.seed + L * 1000 + int(delta * 1e6))

            init_above, lab_above = build_random_batch_at_density(
                rng_build, L, density_above, args.n_random_per_side)
            init_below, lab_below = build_random_batch_at_density(
                rng_build, L, density_below, args.n_random_per_side)
            random_init = np.concatenate([init_above, init_below])
            random_labels = np.concatenate([lab_above, lab_below])

            adv_init, adv_labels, adv_names = build_adversarial_at_density(
                L, density_above, rng_build)

            for dataset, init, labs in [("random", random_init, random_labels),
                                          ("adversarial", adv_init, adv_labels)]:
                rng_run = np.random.default_rng(args.seed * 17 + L + int(delta * 1e6) + len(labs))
                t0 = time.time()
                out = run_strict_ca(
                    init, rule_lut_list=rule_luts, probabilities=probs,
                    max_steps=max_steps, until_consensus=True,
                    backend=backend, rng=rng_run,
                )
                elapsed = time.time() - t0
                L2 = L * L
                correct = _correct_consensus(out["final_totals"], labs, L2)
                times = [t for t in out["time_to_consensus"] if t is not None]
                row = {
                    "mixture": args.mixture,
                    "L": L, "delta": delta,
                    "dataset": dataset,
                    "n_trials": len(labs),
                    "correct_consensus_rate": float(correct.mean()),
                    "n_correct": int(correct.sum()),
                    "n_consensus": len(times),
                    "median_steps_to_consensus": float(np.median(times)) if times else float("inf"),
                    "p95_steps_to_consensus": float(np.percentile(times, 95)) if times else float("inf"),
                    "max_steps_to_consensus": int(max(times)) if times else max_steps,
                    "max_steps_cap": max_steps,
                    "wall_seconds": elapsed,
                }
                runs.append(row)
                hit_cap = len(labs) - len(times)
                print(
                    f"L={L:>3d} d={delta:.3f} {dataset:<12} "
                    f"cc={row['correct_consensus_rate']:.3f} "
                    f"({row['n_correct']}/{row['n_trials']})  "
                    f"med_steps={row['median_steps_to_consensus']:.0f} "
                    f"hit_cap={hit_cap}  "
                    f"[{elapsed:.1f}s]"
                )

    args.output.write_text(json.dumps({
        "mixture": args.mixture,
        "rule_names": rule_names,
        "probabilities": probs,
        "max_steps_mult": args.max_steps_mult,
        "deltas": args.deltas,
        "n_random_per_side": args.n_random_per_side,
        "runs": runs,
    }, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
