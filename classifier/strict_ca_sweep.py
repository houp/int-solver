"""Sweep mixture probabilities for the strict-CA classifier and compare to
the non-local swap baseline.

For each configuration (rule list, probability list, L), run the classifier
on a fixed batch of inputs and report:
- correct-consensus rate,
- median / p95 / max steps-to-consensus,
- wall time.

The non-local swap baseline (scaled_schedule.py with the L-scaled defaults)
is included once per (L) for comparison; the comparison metric is the
ratio of strict-CA median steps to baseline steps (~7L per trial).
"""
from __future__ import annotations

import argparse
import itertools
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

from amplifier_library import apply_radius_step_mlx
from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend
from conservative_noise import apply_lut_mlx, apply_swaps
from strict_ca_classifier import (
    resolve_rule_bits,
    run_strict_ca,
    _correct_consensus,
)
from density_margin_sweep import (
    build_random_batch_at_density,
    build_adversarial_at_density,
)


def baseline_scaled_schedule(
    F_bits: np.ndarray,
    *,
    grid: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    backend,
):
    """Run the non-local swap baseline (scaled schedule from S3) and return
    correct-consensus + nominal-step counts.  Total steps are deterministic:
    7L (CA steps) + 16L swaps = the published baseline cost."""
    T_pre = grid // 2; T_amp = grid; T_shake = grid // 4
    T_amp_final = 4 * grid; K = 8 * grid
    batch, H, W = initial_states.shape
    L2 = H * W
    states = backend.asarray(initial_states, dtype="uint8")
    tiled_F = backend.asarray(np.tile(F_bits, (batch, 1)), dtype="uint8")
    for _ in range(T_pre):
        states = apply_lut_mlx(states, tiled_F)
    for _ in range(2):
        for _ in range(T_amp):
            states = apply_radius_step_mlx(states, "moore81")
        for _ in range(T_shake):
            states = apply_lut_mlx(states, tiled_F)
        s_np = backend.to_numpy(states)
        s_np = apply_swaps(s_np, K, rng)
        states = backend.asarray(s_np, dtype="uint8")
    for _ in range(T_amp_final):
        states = apply_radius_step_mlx(states, "moore81")
    final = backend.to_numpy(states)
    totals = final.reshape(batch, -1).sum(axis=1)
    correct = (
        ((totals == 0) & (labels == 0))
        | ((totals == L2) & (labels == 1))
    )
    return {
        "correct_consensus_rate": float(correct.mean()),
        "n_trials": batch,
        "n_correct": int(correct.sum()),
        "ca_steps_per_trial": T_pre + 2 * (T_amp + T_shake) + T_amp_final,
        "swaps_per_trial": 2 * K,
    }


def evaluate_mixture(
    rule_luts: list[list[int]],
    probabilities: list[float],
    *,
    grid: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    max_steps: int,
    backend,
    rng: np.random.Generator,
) -> dict:
    out = run_strict_ca(
        initial_states,
        rule_lut_list=rule_luts,
        probabilities=probabilities,
        max_steps=max_steps,
        until_consensus=True,
        backend=backend, rng=rng,
        record_density_every=0,
    )
    L2 = grid * grid
    correct = _correct_consensus(out["final_totals"], labels, L2)
    times = [t for t in out["time_to_consensus"] if t is not None]
    return {
        "correct_consensus_rate": float(correct.mean()),
        "n_correct": int(correct.sum()),
        "n_consensus": len(times),
        "n_trials": len(labels),
        "median_steps_to_consensus": float(np.median(times)) if times else float("inf"),
        "p95_steps_to_consensus": float(np.percentile(times, 95)) if times else float("inf"),
        "max_steps_to_consensus": int(max(times)) if times else int(max_steps),
        "max_steps_cap": max_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--baseline-sid", default="sid:58ed6b657afb")
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--delta", type=float, default=0.02,
                        help="Density margin (rho = 0.5 +/- delta)")
    parser.add_argument("--n-random-per-side", type=int, default=16)
    parser.add_argument("--max-steps-mult", type=int, default=64,
                        help="max strict-CA steps = mult * L")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("strict_ca_sweep.json"))

    # mixture configs (each is a list of "rule:prob,rule:prob,...")
    parser.add_argument("--mixture", action="append", required=True,
                        help="Mixture spec: name1:p1,name2:p2,... ; can pass multiple times")
    args = parser.parse_args()

    # Parse mixtures
    mixtures = []
    for spec in args.mixture:
        parts = [seg.strip() for seg in spec.split(",")]
        rule_names = []; probs = []
        for p in parts:
            # Names like "sid:58ed6b657afb" contain a colon, so split from
            # the right to keep the probability separable.
            name, prob = p.rsplit(":", 1)
            rule_names.append(name.strip())
            probs.append(float(prob))
        if abs(sum(probs) - 1.0) > 1e-6:
            raise SystemExit(f"probabilities {probs} for mixture '{spec}' must sum to 1")
        mixtures.append((spec, rule_names, probs))

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    backend = create_backend(args.backend)
    F_baseline = catalog.lut_bits[catalog.resolve_rule_ref(args.baseline_sid)].astype(np.uint8)

    runs = []
    for L in args.grids:
        max_steps = args.max_steps_mult * L
        rng_build = np.random.default_rng(args.seed + L * 1000 + int(args.delta * 1e6))
        density_above = 0.5 + args.delta
        density_below = 0.5 - args.delta
        # Keep the same input batch for all mixtures and the baseline.
        init_above, lab_above = build_random_batch_at_density(
            rng_build, L, density_above, args.n_random_per_side)
        init_below, lab_below = build_random_batch_at_density(
            rng_build, L, density_below, args.n_random_per_side)
        random_init = np.concatenate([init_above, init_below])
        random_labels = np.concatenate([lab_above, lab_below])

        # Baseline (non-local swap)
        rng_bl = np.random.default_rng(args.seed * 7 + L)
        t0 = time.time()
        bl = baseline_scaled_schedule(
            F_baseline, grid=L,
            initial_states=random_init, labels=random_labels,
            rng=rng_bl, backend=backend,
        )
        bl["wall_seconds"] = time.time() - t0
        bl["mixture"] = "baseline_non_local_swap"
        bl["L"] = L
        runs.append(bl)
        print(f"L={L:>3d}  BASELINE  cc={bl['correct_consensus_rate']:.3f}  "
              f"steps/trial={bl['ca_steps_per_trial']} CA + {bl['swaps_per_trial']} swaps "
              f"[{bl['wall_seconds']:.1f}s]")

        # Mixtures
        for spec, names, probs in mixtures:
            rule_luts = [resolve_rule_bits(n, catalog=catalog) for n in names]
            rng_mix = np.random.default_rng(args.seed * 13 + L + hash(spec) % (1 << 31))
            t0 = time.time()
            r = evaluate_mixture(
                rule_luts, probs, grid=L,
                initial_states=random_init, labels=random_labels,
                max_steps=max_steps,
                backend=backend, rng=rng_mix,
            )
            r["wall_seconds"] = time.time() - t0
            r["mixture"] = spec
            r["L"] = L
            r["delta"] = args.delta
            runs.append(r)
            speedup_ratio = (
                r["median_steps_to_consensus"] / bl["ca_steps_per_trial"]
                if r["median_steps_to_consensus"] != float("inf") else float("inf")
            )
            print(f"L={L:>3d}  {spec[:40]:<40}  cc={r['correct_consensus_rate']:.3f}  "
                  f"med_steps={r['median_steps_to_consensus']:.0f} "
                  f"({speedup_ratio:.1f}x baseline)  "
                  f"[{r['wall_seconds']:.1f}s]")

    args.output.write_text(json.dumps({
        "delta": args.delta,
        "n_random_per_side": args.n_random_per_side,
        "max_steps_mult": args.max_steps_mult,
        "runs": runs,
    }, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
