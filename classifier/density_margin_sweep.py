"""S1: Density-margin scaling law.

For each (L, delta) where delta = |rho - 1/2|, run the L-scaled stochastic
classifier on:
- random inputs at exact density rho = 0.5 +/- delta (shuffled fixed-density
  binary vector, so realized density is exact),
- the adversarial structured battery (stripes, checker, block_checker,
  half_half) at the same density.

The aim is to map out rho_min(L): the smallest |rho - 1/2| at which the
schedule still gives 100% correct consensus.

Theoretical prediction: rho_min(L) ~ Theta(1/L^2), since at smaller bias
the global majority signal falls below the per-cell scale.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# --- repo-root path bootstrap (keep BEFORE ca_search/local imports) ---
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
_CATALOG_DIR = _REPO_ROOT / "catalogs"
# ---------------------------------------------------------------------

from amplifier_library import apply_radius_step_mlx
from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend
from conservative_noise import apply_lut_mlx, apply_swaps
from adversarial_realistic import (
    make_stripe_with_density,
    make_checker_with_density,
    make_block_checker_with_density,
    make_half_half_with_density,
)


def make_random_with_exact_density(L: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Build an L x L binary lattice with exactly round(density * L^2) ones,
    placed uniformly at random."""
    n_total = L * L
    n_ones = int(round(density * n_total))
    flat = np.zeros(n_total, dtype=np.uint8)
    flat[:n_ones] = 1
    rng.shuffle(flat)
    return flat.reshape(L, L)


def build_random_batch_at_density(rng, L: int, target_density: float, n_trials: int) -> tuple[np.ndarray, np.ndarray]:
    inits = []
    for _ in range(n_trials):
        inits.append(make_random_with_exact_density(L, target_density, rng))
    init = np.stack(inits)
    actual_n_ones = init.reshape(n_trials, -1).sum(axis=1)
    target_label = 1 if target_density > 0.5 else 0
    labels = np.full(n_trials, target_label, dtype=np.uint8)
    return init, labels


def build_adversarial_at_density(L: int, density: float, rng: np.random.Generator):
    """5 patterns x both labels yields 10 cases at the given density.
    For a structured pattern at density d, the actual majority label is
    1 if d > 0.5 else 0. We test both d=density (label=1 if density>0.5)
    and d=1-density (label is flipped).
    """
    cases, names, labels = [], [], []
    for actual_d in [density, 1.0 - density]:
        target_label = 1 if actual_d > 0.5 else 0
        for name, fn in [
            ("stripes_h", lambda r=rng, d=actual_d: make_stripe_with_density(L, "h", d, r)),
            ("stripes_v", lambda r=rng, d=actual_d: make_stripe_with_density(L, "v", d, r)),
            ("checker", lambda r=rng, d=actual_d: make_checker_with_density(L, d, r)),
            ("block_checker", lambda r=rng, d=actual_d: make_block_checker_with_density(L, d, r)),
            ("half_half", lambda r=rng, d=actual_d: make_half_half_with_density(L, d, r)),
        ]:
            s = fn()
            cases.append(s.copy()); names.append(f"{name}_d{actual_d:.4f}"); labels.append(target_label)
    return np.stack(cases), np.asarray(labels, dtype=np.uint8), names


def run_scaled_schedule(
    F_bits: np.ndarray,
    *,
    grid: int,
    c_pre: float, c_amp: float, c_shake: float, c_final: float, k_swap: float,
    num_shakes: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    backend,
) -> dict:
    T_pre = int(c_pre * grid)
    T_amp = int(c_amp * grid)
    T_shake = int(c_shake * grid)
    T_amp_final = int(c_final * grid)
    K = int(k_swap * grid)
    batch, H, W = initial_states.shape
    L2 = H * W
    states = backend.asarray(initial_states, dtype="uint8")
    tiled_F = backend.asarray(np.tile(F_bits, (batch, 1)), dtype="uint8")

    for _ in range(T_pre):
        states = apply_lut_mlx(states, tiled_F)
    for _ in range(num_shakes):
        for _ in range(T_amp):
            states = apply_radius_step_mlx(states, "moore81")
        for _ in range(T_shake):
            states = apply_lut_mlx(states, tiled_F)
        if K > 0:
            s_np = backend.to_numpy(states)
            s_np = apply_swaps(s_np, K, rng)
            states = backend.asarray(s_np, dtype="uint8")
    for _ in range(T_amp_final):
        states = apply_radius_step_mlx(states, "moore81")

    final = backend.to_numpy(states)
    totals = final.reshape(batch, -1).sum(axis=1)
    all_zero = (totals == 0)
    all_one = (totals == L2)
    consensus = all_zero | all_one
    correct = (all_zero & (labels == 0)) | (all_one & (labels == 1))
    return {
        "consensus_rate": float(consensus.mean()),
        "correct_consensus_rate": float(correct.mean()),
        "num_failures": int((~correct).sum()),
        "trials": int(batch),
        "per_trial_correct": correct.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sid", default="sid:58ed6b657afb")
    # Schedule parameters
    parser.add_argument("--c-pre", type=float, default=0.5)
    parser.add_argument("--c-amp", type=float, default=1.0)
    parser.add_argument("--c-shake", type=float, default=0.25)
    parser.add_argument("--c-final", type=float, default=4.0)
    parser.add_argument("--k-swap", type=float, default=8.0)
    parser.add_argument("--num-shakes", type=int, default=2)
    # Sweep parameters
    parser.add_argument("--grids", type=int, nargs="+",
                        default=[64, 128, 192, 256, 384, 512])
    parser.add_argument("--deltas", type=float, nargs="+",
                        default=[0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
                        help="|rho - 1/2| values to test")
    parser.add_argument("--random-trials-per-side", type=int, default=64,
                        help="Random trials at each (L, delta, side); halved for the "
                             "two L values that make the wall-clock grid-quadrant huge")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path,
                        default=Path("density_margin_sweep.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    F_bits = catalog.lut_bits[catalog.resolve_rule_ref(args.sid)].astype(np.uint8)
    backend = create_backend(args.backend)

    runs = []
    summary_table = []

    for grid in args.grids:
        # Halve trial count for huge grids to keep total wall time manageable
        n_random = args.random_trials_per_side
        if grid >= 384:
            n_random = max(16, n_random // 2)
        if grid >= 1024:
            n_random = max(8, n_random // 4)

        for delta in args.deltas:
            # Two sides: rho = 0.5 + delta (label 1) and rho = 0.5 - delta (label 0)
            density_above = 0.5 + delta
            density_below = 0.5 - delta

            # Cap delta where the rounded particle count would not actually
            # break the tie (for very small delta and grid)
            n_total = grid * grid
            actual_above = int(round(density_above * n_total))
            actual_below = int(round(density_below * n_total))
            if actual_above <= n_total // 2 or actual_below >= n_total // 2:
                # Tie or wrong-side rounding; skip
                summary_table.append({
                    "grid": grid, "delta": delta,
                    "skipped": True,
                    "reason": "rounded particle count does not break the tie"
                })
                continue

            t0 = time.time()

            # Random sweep
            rng_random = np.random.default_rng(args.seed + grid * 1000 + int(delta * 1e6))
            init_above, lab_above = build_random_batch_at_density(rng_random, grid, density_above, n_random)
            init_below, lab_below = build_random_batch_at_density(rng_random, grid, density_below, n_random)
            random_init = np.concatenate([init_above, init_below])
            random_labels = np.concatenate([lab_above, lab_below])
            rng_sched = np.random.default_rng(args.seed + grid * 7919 + int(delta * 1e6))
            r_random = run_scaled_schedule(
                F_bits, grid=grid,
                c_pre=args.c_pre, c_amp=args.c_amp, c_shake=args.c_shake,
                c_final=args.c_final, k_swap=args.k_swap, num_shakes=args.num_shakes,
                initial_states=random_init, labels=random_labels,
                rng=rng_sched, backend=backend,
            )

            # Adversarial sweep
            rng_adv_build = np.random.default_rng(args.seed + grid * 6151 + int(delta * 1e6))
            adv_init, adv_labels, adv_names = build_adversarial_at_density(grid, density_above, rng_adv_build)
            rng_sched_adv = np.random.default_rng(args.seed + grid * 6151 + int(delta * 1e6) + 1)
            r_adv = run_scaled_schedule(
                F_bits, grid=grid,
                c_pre=args.c_pre, c_amp=args.c_amp, c_shake=args.c_shake,
                c_final=args.c_final, k_swap=args.k_swap, num_shakes=args.num_shakes,
                initial_states=adv_init, labels=adv_labels,
                rng=rng_sched_adv, backend=backend,
            )

            elapsed = time.time() - t0
            row = {
                "grid": grid,
                "delta": delta,
                "density_above": density_above,
                "density_below": density_below,
                "n_random_per_side": n_random,
                "random_cc": r_random["correct_consensus_rate"],
                "random_fails": r_random["num_failures"],
                "random_trials": r_random["trials"],
                "adversarial_cc": r_adv["correct_consensus_rate"],
                "adversarial_fails": r_adv["num_failures"],
                "adversarial_trials": r_adv["trials"],
                "elapsed_seconds": elapsed,
            }
            adv_per_case = []
            for i, name in enumerate(adv_names):
                adv_per_case.append({
                    "name": name,
                    "label": int(adv_labels[i]),
                    "correct": bool(r_adv["per_trial_correct"][i]),
                })
            row["adversarial_cases"] = adv_per_case
            summary_table.append(row)
            runs.append(row)

            print(f"L={grid:>4d} delta={delta:<6.4f} "
                  f"random_cc={r_random['correct_consensus_rate']:.4f} "
                  f"({r_random['num_failures']:>3d}/{r_random['trials']}) "
                  f"adv_cc={r_adv['correct_consensus_rate']:.4f} "
                  f"({r_adv['num_failures']:>3d}/{r_adv['trials']}) "
                  f"[{elapsed:.1f}s]")

    # Compute rho_min(L): smallest delta for which both random and adversarial
    # are 100% correct at that grid size.
    rho_min_per_grid: dict[int, float] = {}
    for grid in args.grids:
        for delta in sorted(args.deltas, reverse=True):
            row = next((r for r in runs if r["grid"] == grid and r["delta"] == delta), None)
            if row is None:
                continue
            if row["random_cc"] == 1.0 and row["adversarial_cc"] == 1.0:
                rho_min_per_grid[grid] = delta
        if grid not in rho_min_per_grid:
            rho_min_per_grid[grid] = float("inf")

    args.output.write_text(json.dumps({
        "schedule_params": {
            "c_pre": args.c_pre, "c_amp": args.c_amp,
            "c_shake": args.c_shake, "c_final": args.c_final,
            "k_swap": args.k_swap, "num_shakes": args.num_shakes,
            "sid": args.sid,
        },
        "deltas": args.deltas,
        "grids": args.grids,
        "runs": runs,
        "summary_table": summary_table,
        "rho_min_per_grid": rho_min_per_grid,
    }, indent=2))
    print(f"\nrho_min(L) summary (smallest delta that gives 100% on both random and adv):")
    for grid in args.grids:
        m = rho_min_per_grid.get(grid, float("inf"))
        print(f"  L={grid:>4d}: rho_min = {m:.5g}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
