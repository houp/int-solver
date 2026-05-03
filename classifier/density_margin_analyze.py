"""Analyze density_margin_sweep.json:
- Print a (L x delta) success-rate table.
- Compute rho_min(L) for both random and adversarial datasets.
- Fit log(rho_min) ~ alpha * log(L) + beta and report the scaling exponent.
- Identify which adversarial cases are bottlenecks at the failure boundary.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="*",
                        default=[Path("results/density_classification/2026-04-27/density_margin_sweep.json")],
                        help="One or more sweep JSON files; runs are concatenated")
    args = parser.parse_args()

    runs = []
    for p in args.paths:
        data = json.loads(p.read_text())
        runs.extend(data["runs"])
    # Dedup by (grid, delta) — last write wins (later files override)
    deduped = {}
    for r in runs:
        deduped[(r["grid"], r["delta"])] = r
    runs = list(deduped.values())

    grids = sorted({r["grid"] for r in runs})
    deltas = sorted({r["delta"] for r in runs}, reverse=True)

    # === Table 1: random correct-consensus rate ===
    print("=== Random correct-consensus rate ===")
    header = "       " + "  ".join(f"d={d:.4f}" for d in deltas)
    print(header)
    for L in grids:
        row = [f"L={L:>4d}:"]
        for d in deltas:
            r = next((r for r in runs if r["grid"] == L and r["delta"] == d), None)
            if r is None:
                row.append("  --   ")
            else:
                row.append(f" {r['random_cc']:.4f}")
        print(" ".join(row))

    # === Table 2: adversarial correct-consensus rate ===
    print("\n=== Adversarial correct-consensus rate ===")
    print(header)
    for L in grids:
        row = [f"L={L:>4d}:"]
        for d in deltas:
            r = next((r for r in runs if r["grid"] == L and r["delta"] == d), None)
            if r is None:
                row.append("  --   ")
            else:
                row.append(f" {r['adversarial_cc']:.4f}")
        print(" ".join(row))

    # === rho_min for both datasets ===
    print("\n=== rho_min(L) (smallest delta with 100% correct consensus) ===")
    print("       random       adversarial   joint(min of both)")
    rho_min_random = {}
    rho_min_adv = {}
    rho_min_joint = {}
    for L in grids:
        for d in sorted(deltas, reverse=True):
            r = next((r for r in runs if r["grid"] == L and r["delta"] == d), None)
            if r is None:
                continue
            if r["random_cc"] == 1.0:
                rho_min_random[L] = d
            if r["adversarial_cc"] == 1.0:
                rho_min_adv[L] = d
            if r["random_cc"] == 1.0 and r["adversarial_cc"] == 1.0:
                rho_min_joint[L] = d
        print(f"L={L:>4d}:  "
              f"{rho_min_random.get(L, float('nan')):.5g}    "
              f"{rho_min_adv.get(L, float('nan')):.5g}    "
              f"{rho_min_joint.get(L, float('nan')):.5g}")

    # === Scaling fit: log(rho_min) ~ alpha * log(L) + beta ===
    print("\n=== Scaling fit ===")
    for label, dct in [("random", rho_min_random), ("adversarial", rho_min_adv), ("joint", rho_min_joint)]:
        Ls = [L for L in grids if dct.get(L, math.inf) != math.inf]
        if len(Ls) < 2:
            print(f"{label}: not enough data points")
            continue
        xs = np.log(Ls, dtype=np.float64)
        ys = np.log([dct[L] for L in Ls], dtype=np.float64)
        # least-squares fit
        A = np.vstack([xs, np.ones_like(xs)]).T
        slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
        # report constant: rho_min ~ C * L^slope
        C = math.exp(intercept)
        print(f"{label:<13}: rho_min(L) ~ {C:.4g} * L^{slope:.3f}")
        # Compare to 1/L^2 prediction
        if abs(slope - (-2.0)) < 0.5:
            print(f"             (theoretical ~ 1/L^2 = L^-2; fit slope = {slope:.3f})")

    # === Bottleneck adversarial cases at the failure boundary ===
    print("\n=== Bottleneck adversarial cases at the failure edge ===")
    for L in grids:
        # Find the delta level just below rho_min where adversarial fails
        for d in sorted(deltas):
            r = next((r for r in runs if r["grid"] == L and r["delta"] == d), None)
            if r is None or r["adversarial_cc"] == 1.0:
                continue
            cases = r.get("adversarial_cases", [])
            failing = [c["name"] for c in cases if not c["correct"]]
            if failing:
                print(f"  L={L:>4d} delta={d:.4f}: {len(failing)}/{len(cases)} adv fails")
                kinds = {}
                for n in failing:
                    base = n.rsplit("_d", 1)[0]
                    kinds[base] = kinds.get(base, 0) + 1
                for base, count in sorted(kinds.items(), key=lambda x: -x[1]):
                    print(f"      {base}: {count}")
                break  # Just the first failing delta

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
