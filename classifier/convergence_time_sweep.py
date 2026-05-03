"""Empirical (L, delta) sweep of time-to-consensus for the rebalanced
strict-CA mixture, to test Conjecture (two-phase log + L absorption)
from Section 5.13 of the technical report.

For each (L, delta) cell we:
  - generate n_seeds random Bernoulli(1/2 + delta) configurations,
  - simulate the strict-CA mixture until consensus,
  - record absorption time tau and whether it absorbed at the correct
    consensus.

Output: a JSON table of all (L, delta, seed) -> (tau, correct) records,
plus aggregate statistics per cell.

Conjecture under test:
  E[tau] = alpha log(1/delta) + beta L + gamma
  Pr(correct) >= 1 - exp(-c L delta^2)

Default grid: L in {64, 96, 128, 192}, delta in {0.01, 0.02, 0.05,
0.1, 0.2}, n_seeds=32. Wall-time estimate at default: roughly 30-60 min
on M2 Max MLX.
"""
from __future__ import annotations

import argparse
import json
import sys
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
from density_margin_sweep import build_random_batch_at_density


REBALANCED_RULES = [
    "sid:58ed6b657afb",
    "traffic_ne", "traffic_nw", "traffic_se", "traffic_sw",
    "moore_maj",
]
REBALANCED_WEIGHTS = [0.041, 0.049, 0.126, 0.017, 0.579, 0.188]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ls", type=int, nargs="+", default=[64, 96, 128, 192])
    parser.add_argument("--deltas", type=float, nargs="+",
                        default=[0.01, 0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--n-seeds", type=int, default=32,
                        help="Trials per (L, delta) cell, both labels combined")
    parser.add_argument("--max-steps-mult", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path,
                        default=Path("results/density_classification/2026-05-01/convergence_time_sweep.json"))
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    catalog = load_binary_catalog(
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.bin"),
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.json"),
    )
    rule_lut_list = [resolve_rule_bits(name, catalog=catalog) for name in REBALANCED_RULES]
    backend = create_backend("mlx")

    print(f"Sweep grid: Ls={args.Ls} x deltas={args.deltas} x n_seeds={args.n_seeds}")
    print(f"Rules: {REBALANCED_RULES}")
    print(f"Weights: {REBALANCED_WEIGHTS}")
    print()

    all_records = []
    aggregate = {}
    t0 = time.time()

    for L in args.Ls:
        max_steps = args.max_steps_mult * L
        for delta in args.deltas:
            t_cell = time.time()
            rng_build = np.random.default_rng(args.seed * 7919 + int(L * 1000 + delta * 1e6))
            n_per_side = max(1, args.n_seeds // 2)
            init_a, lab_a = build_random_batch_at_density(
                rng_build, L, 0.5 + delta, n_per_side)
            init_b, lab_b = build_random_batch_at_density(
                rng_build, L, 0.5 - delta, n_per_side)
            inits = np.concatenate([init_a, init_b])
            labels = np.concatenate([lab_a, lab_b])

            rng_run = np.random.default_rng(args.seed * 31 + int(L * 1000 + delta * 1e6))
            out = run_strict_ca(
                inits, rule_lut_list=rule_lut_list,
                probabilities=REBALANCED_WEIGHTS,
                max_steps=max_steps, until_consensus=True,
                backend=backend, rng=rng_run,
            )
            correct = _correct_consensus(out["final_totals"], labels, L * L)
            times = out["time_to_consensus"]
            times_finite = [t for t in times if t is not None]

            for k in range(len(labels)):
                all_records.append({
                    "L": L, "delta": delta, "seed_idx": k,
                    "label": int(labels[k]),
                    "tau": (int(times[k]) if times[k] is not None else None),
                    "correct": bool(correct[k]),
                })

            n_total = len(labels)
            n_correct = int(correct.sum())
            n_consensus = len(times_finite)
            tau_med = float(np.median(times_finite)) if times_finite else float("nan")
            tau_mean = float(np.mean(times_finite)) if times_finite else float("nan")
            tau_p90 = float(np.percentile(times_finite, 90)) if times_finite else float("nan")
            elapsed = time.time() - t_cell

            aggregate[f"L{L}_d{delta}"] = {
                "L": L, "delta": delta,
                "n_total": n_total, "n_correct": n_correct,
                "n_consensus": n_consensus,
                "cc": n_correct / n_total,
                "tau_median": tau_med, "tau_mean": tau_mean, "tau_p90": tau_p90,
                "elapsed_seconds": elapsed,
            }
            print(f"  L={L:>3}  delta={delta:.3f}   "
                  f"cc={n_correct}/{n_total} ({n_correct / n_total:.4f})   "
                  f"tau median={tau_med:.0f} mean={tau_mean:.0f} p90={tau_p90:.0f}   "
                  f"[{elapsed:.1f}s]")

    print()
    print(f"=== Aggregate (total {time.time() - t0:.1f}s) ===")
    print(f"{'L':>4} {'delta':>7} {'cc':>8} {'tau_med':>8} {'tau_mean':>8}")
    for key, agg in aggregate.items():
        print(f"{agg['L']:>4} {agg['delta']:>7.3f} {agg['cc']:>8.4f} "
              f"{agg['tau_median']:>8.0f} {agg['tau_mean']:>8.0f}")

    # Fit alpha log(1/delta) + beta L + gamma to tau_mean
    Ls = np.array([agg["L"] for agg in aggregate.values()])
    ds = np.array([agg["delta"] for agg in aggregate.values()])
    taus = np.array([agg["tau_mean"] for agg in aggregate.values()])
    mask = np.isfinite(taus)
    if mask.sum() >= 3:
        X = np.column_stack([np.log(1.0 / ds[mask]), Ls[mask], np.ones(mask.sum())])
        coef, *_ = np.linalg.lstsq(X, taus[mask], rcond=None)
        alpha, beta, gamma = coef
        pred = X @ coef
        ss_res = float(((taus[mask] - pred) ** 2).sum())
        ss_tot = float(((taus[mask] - taus[mask].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print()
        print(f"=== OLS fit: E[tau] = alpha log(1/delta) + beta L + gamma ===")
        print(f"  alpha = {alpha:.3f}")
        print(f"  beta  = {beta:.3f}")
        print(f"  gamma = {gamma:.3f}")
        print(f"  R^2   = {r2:.4f}")
        fit = {"alpha": float(alpha), "beta": float(beta),
               "gamma": float(gamma), "R2": float(r2),
               "n_cells_fit": int(mask.sum())}
    else:
        fit = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "args": vars(args) | {"output": str(args.output)},
        "rules": REBALANCED_RULES,
        "weights": REBALANCED_WEIGHTS,
        "aggregate": aggregate,
        "fit_log_delta_plus_L": fit,
        "all_records": all_records,
        "elapsed_total_seconds": time.time() - t0,
    }, indent=2, default=str))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
