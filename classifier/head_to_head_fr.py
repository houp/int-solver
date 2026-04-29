"""Head-to-head comparison between our scaled stochastic classifier and the
Fates-Regnault (2016) construction on identical initial configurations.

For each grid size L and each input set:
- generate the same initial configurations once;
- run both classifiers on them;
- record correct-consensus rate, wall time, and (if you supply per-trial
  budgets) the number of CA steps consumed by each.

We compare on:
  RND : random Bernoulli(rho) with rho in {0.49, 0.51}, exact density.
  ADV : 10-case structured-adversarial battery (stripes_h/v, checker,
        block_checker, half_half) at densities {0.51, 0.55} with both labels.
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
from fates_regnault_2016 import run_fates_regnault
from density_margin_sweep import (
    build_random_batch_at_density,
    build_adversarial_at_density,
    run_scaled_schedule,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sid", default="sid:58ed6b657afb")
    parser.add_argument("--grids", type=int, nargs="+", default=[50, 64, 100, 128, 192, 256])
    parser.add_argument("--rho-margins", type=float, nargs="+", default=[0.01, 0.05])
    parser.add_argument("--n-random-per-side", type=int, default=32)
    # Fates-Regnault schedule lengths follow the paper: T1=T2 ~ 3 * L^2.
    # The user can override with --fr-T-mult.
    parser.add_argument("--fr-T-mult", type=float, default=3.0,
                        help="Fates-Regnault T1=T2 = fr_T_mult * L^2")
    # Our scaled schedule
    parser.add_argument("--c-pre", type=float, default=0.5)
    parser.add_argument("--c-amp", type=float, default=1.0)
    parser.add_argument("--c-shake", type=float, default=0.25)
    parser.add_argument("--c-final", type=float, default=4.0)
    parser.add_argument("--k-swap", type=float, default=8.0)
    parser.add_argument("--num-shakes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("head_to_head_fr.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    F_bits = catalog.lut_bits[catalog.resolve_rule_ref(args.sid)].astype(np.uint8)
    backend = create_backend(args.backend)

    runs = []

    for L in args.grids:
        T_fr = max(1, int(round(args.fr_T_mult * L * L)))
        # Step counts (deterministic part of our schedule) for fairness
        ours_steps = (
            int(args.c_pre * L)
            + args.num_shakes * (int(args.c_amp * L) + int(args.c_shake * L))
            + int(args.c_final * L)
        )

        for delta in args.rho_margins:
            density_above = 0.5 + delta
            density_below = 0.5 - delta
            n_total = L * L
            actual_above = int(round(density_above * n_total))
            actual_below = int(round(density_below * n_total))
            if actual_above <= n_total // 2 or actual_below >= n_total // 2:
                continue

            rng_build = np.random.default_rng(args.seed + L * 1000 + int(delta * 1e6))

            # Random inputs
            init_above, lab_above = build_random_batch_at_density(
                rng_build, L, density_above, args.n_random_per_side)
            init_below, lab_below = build_random_batch_at_density(
                rng_build, L, density_below, args.n_random_per_side)
            random_init = np.concatenate([init_above, init_below])
            random_labels = np.concatenate([lab_above, lab_below])

            # Adversarial inputs
            adv_init, adv_labels, adv_names = build_adversarial_at_density(
                L, density_above, rng_build)

            # ---- our classifier ----
            rng_ours_rand = np.random.default_rng(args.seed * 7 + L * 1000 + int(delta * 1e6))
            t0 = time.time()
            r_ours_rand = run_scaled_schedule(
                F_bits, grid=L,
                c_pre=args.c_pre, c_amp=args.c_amp, c_shake=args.c_shake,
                c_final=args.c_final, k_swap=args.k_swap, num_shakes=args.num_shakes,
                initial_states=random_init, labels=random_labels,
                rng=rng_ours_rand, backend=backend,
            )
            t_ours_rand = time.time() - t0
            rng_ours_adv = np.random.default_rng(args.seed * 11 + L * 1000 + int(delta * 1e6))
            t0 = time.time()
            r_ours_adv = run_scaled_schedule(
                F_bits, grid=L,
                c_pre=args.c_pre, c_amp=args.c_amp, c_shake=args.c_shake,
                c_final=args.c_final, k_swap=args.k_swap, num_shakes=args.num_shakes,
                initial_states=adv_init, labels=adv_labels,
                rng=rng_ours_adv, backend=backend,
            )
            t_ours_adv = time.time() - t0

            # ---- Fates-Regnault classifier ----
            rng_fr_rand = np.random.default_rng(args.seed * 13 + L * 1000 + int(delta * 1e6))
            t0 = time.time()
            fr_rand_final, _ = run_fates_regnault(
                random_init, T1=T_fr, T2=T_fr, rng=rng_fr_rand)
            t_fr_rand = time.time() - t0
            fr_rand_totals = fr_rand_final.reshape(len(random_labels), -1).sum(axis=1)
            fr_rand_correct = (
                ((fr_rand_totals == 0) & (random_labels == 0))
                | ((fr_rand_totals == L * L) & (random_labels == 1))
            )
            fr_rand_cc = float(fr_rand_correct.mean())

            rng_fr_adv = np.random.default_rng(args.seed * 17 + L * 1000 + int(delta * 1e6))
            t0 = time.time()
            fr_adv_final, _ = run_fates_regnault(
                adv_init, T1=T_fr, T2=T_fr, rng=rng_fr_adv)
            t_fr_adv = time.time() - t0
            fr_adv_totals = fr_adv_final.reshape(len(adv_labels), -1).sum(axis=1)
            fr_adv_correct = (
                ((fr_adv_totals == 0) & (adv_labels == 0))
                | ((fr_adv_totals == L * L) & (adv_labels == 1))
            )
            fr_adv_cc = float(fr_adv_correct.mean())

            row = {
                "L": L,
                "delta": delta,
                "density_above": density_above,
                "n_random_per_side": args.n_random_per_side,
                "n_random_total": len(random_labels),
                "n_adv": len(adv_labels),
                # ours
                "ours_random_cc": r_ours_rand["correct_consensus_rate"],
                "ours_random_fails": r_ours_rand["num_failures"],
                "ours_adv_cc": r_ours_adv["correct_consensus_rate"],
                "ours_adv_fails": r_ours_adv["num_failures"],
                "ours_steps_per_trial": ours_steps,
                "ours_swaps_total": int(args.k_swap * L) * args.num_shakes,
                "ours_random_seconds": t_ours_rand,
                "ours_adv_seconds": t_ours_adv,
                # FR
                "fr_T1": T_fr,
                "fr_T2": T_fr,
                "fr_steps_per_trial": 2 * T_fr,
                "fr_random_cc": fr_rand_cc,
                "fr_random_fails": int((~fr_rand_correct).sum()),
                "fr_adv_cc": fr_adv_cc,
                "fr_adv_fails": int((~fr_adv_correct).sum()),
                "fr_random_seconds": t_fr_rand,
                "fr_adv_seconds": t_fr_adv,
                # adversarial per-case detail
                "fr_adv_per_case": [
                    {"name": n, "label": int(adv_labels[i]), "correct": bool(fr_adv_correct[i])}
                    for i, n in enumerate(adv_names)
                ],
                "ours_adv_per_case": [
                    {"name": n, "label": int(adv_labels[i]),
                     "correct": bool(r_ours_adv["per_trial_correct"][i])}
                    for i, n in enumerate(adv_names)
                ],
            }
            runs.append(row)
            print(
                f"L={L:>4d} delta={delta:<5.3f}  "
                f"ours: rand={row['ours_random_cc']:.3f} ({row['ours_random_fails']:>2d}/{row['n_random_total']}) "
                f"adv={row['ours_adv_cc']:.3f} ({row['ours_adv_fails']}/{row['n_adv']})  "
                f"FR: rand={row['fr_random_cc']:.3f} ({row['fr_random_fails']:>2d}/{row['n_random_total']}) "
                f"adv={row['fr_adv_cc']:.3f} ({row['fr_adv_fails']}/{row['n_adv']})  "
                f"steps: ours={ours_steps:>5d}, FR={2*T_fr:>6d}  "
                f"[ours: {t_ours_rand+t_ours_adv:.1f}s, FR: {t_fr_rand+t_fr_adv:.1f}s]"
            )

    args.output.write_text(json.dumps({
        "schedule_params": {
            "ours": {
                "sid": args.sid,
                "c_pre": args.c_pre, "c_amp": args.c_amp,
                "c_shake": args.c_shake, "c_final": args.c_final,
                "k_swap": args.k_swap, "num_shakes": args.num_shakes,
            },
            "fr": {"T_mult": args.fr_T_mult},
        },
        "runs": runs,
    }, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
