"""When is the final M^{4L} amplifier actually needed?

For each input we step through the L-scaled schedule and record the
*earliest* phase boundary at which consensus (rho in {0, 1}) was already
reached.  This tells us how much of the final amplification is wasted on
typical inputs and how much is required on hard inputs.

We classify each trial into one of:
  - 'after_F_pre'        : consensus already after F^{T_pre}
                           (only possible if input was already near-consensus)
  - 'after_amp{i}'        : consensus after the amplifier of shake i
  - 'after_shake{i}'      : consensus after F + swap of shake i
  - 'after_final_amp'     : consensus only after the final M^{4L}
  - 'never_reached'       : did not reach consensus at all

For 'after_final_amp' and 'never_reached' the final M^{4L} is doing real
work (or the schedule is failing).  For everything else the final amp is
redundant on that trial and could be skipped.
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
from density_margin_sweep import (
    build_random_batch_at_density,
    build_adversarial_at_density,
)


def classify_trial(consensus_history: list[tuple[str, np.ndarray]], L2: int) -> tuple[str, str | None]:
    """Given a sequence of (label, density_array) checkpoints in execution
    order, return (earliest_consensus_label, final_consensus).
    'final_consensus' is one of {'all_zero', 'all_one', None}.
    """
    earliest = None
    for label, totals in consensus_history:
        for t in totals:
            if t == 0 or t == L2:
                pass  # consensus reached for at least one trial; not relevant for per-trial
        # We're called per-trial with totals being a single int; restructure outside
    return None, None  # unused, see below


def study_inputs(
    F_bits: np.ndarray,
    *,
    grid: int,
    c_pre: float, c_amp: float, c_shake: float, c_final: float, k_swap: float,
    num_shakes: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    backend,
):
    T_pre = int(c_pre * grid); T_amp = int(c_amp * grid)
    T_shake = int(c_shake * grid); T_amp_final = int(c_final * grid)
    K = int(k_swap * grid)
    batch, H, W = initial_states.shape
    L2 = H * W

    states = backend.asarray(initial_states, dtype="uint8")
    tiled_F = backend.asarray(np.tile(F_bits, (batch, 1)), dtype="uint8")

    # Track for each trial the earliest phase boundary at which consensus
    # was reached (matching the correct global majority).
    earliest = ["never_reached"] * batch
    consensus_at_final_only = [False] * batch
    last_correct = [False] * batch

    def update(phase_label: str):
        cur = backend.to_numpy(states)
        totals = cur.reshape(batch, -1).sum(axis=1)
        for i in range(batch):
            if earliest[i] != "never_reached":
                continue
            t = int(totals[i])
            if t == 0 and labels[i] == 0:
                earliest[i] = phase_label
            elif t == L2 and labels[i] == 1:
                earliest[i] = phase_label
        return totals

    # Phase 0: initial
    update("initial")  # most likely no-op; sanity
    # Phase 1: F^{T_pre}
    for _ in range(T_pre):
        states = apply_lut_mlx(states, tiled_F)
    update("after_F_pre")

    # Shake cycles
    for s_idx in range(num_shakes):
        for _ in range(T_amp):
            states = apply_radius_step_mlx(states, "moore81")
        update(f"after_amp{s_idx + 1}")
        for _ in range(T_shake):
            states = apply_lut_mlx(states, tiled_F)
        update(f"after_shake_F{s_idx + 1}")
        if K > 0:
            s_np = backend.to_numpy(states)
            s_np = apply_swaps(s_np, K, rng)
            states = backend.asarray(s_np, dtype="uint8")
        update(f"after_shake_swap{s_idx + 1}")

    # Final amp
    for _ in range(T_amp_final):
        states = apply_radius_step_mlx(states, "moore81")
    update("after_final_amp")

    # Final correctness
    final_totals = backend.to_numpy(states).reshape(batch, -1).sum(axis=1)
    for i in range(batch):
        last_correct[i] = bool(
            (final_totals[i] == 0 and labels[i] == 0)
            or (final_totals[i] == L2 and labels[i] == 1)
        )

    return earliest, last_correct


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sid", default="sid:58ed6b657afb")
    parser.add_argument("--c-pre", type=float, default=0.5)
    parser.add_argument("--c-amp", type=float, default=1.0)
    parser.add_argument("--c-shake", type=float, default=0.25)
    parser.add_argument("--c-final", type=float, default=4.0)
    parser.add_argument("--k-swap", type=float, default=8.0)
    parser.add_argument("--num-shakes", type=int, default=2)
    parser.add_argument("--grids", type=int, nargs="+", default=[128, 192, 256])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.02, 0.01])
    parser.add_argument("--n-random-per-side", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("early_stop_study.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    F_bits = catalog.lut_bits[catalog.resolve_rule_ref(args.sid)].astype(np.uint8)
    backend = create_backend(args.backend)

    runs = []

    for L in args.grids:
        for delta in args.deltas:
            density_above = 0.5 + delta
            density_below = 0.5 - delta
            n_total = L * L
            actual_above = int(round(density_above * n_total))
            actual_below = int(round(density_below * n_total))
            if actual_above <= n_total // 2 or actual_below >= n_total // 2:
                continue

            rng_build = np.random.default_rng(args.seed + L * 1000 + int(delta * 1e6))

            # Random
            init_above, lab_above = build_random_batch_at_density(
                rng_build, L, density_above, args.n_random_per_side)
            init_below, lab_below = build_random_batch_at_density(
                rng_build, L, density_below, args.n_random_per_side)
            random_init = np.concatenate([init_above, init_below])
            random_labels = np.concatenate([lab_above, lab_below])

            # Adversarial
            adv_init, adv_labels, adv_names = build_adversarial_at_density(
                L, density_above, rng_build)

            for dataset_name, inits, labs in [("random", random_init, random_labels),
                                                ("adversarial", adv_init, adv_labels)]:
                rng_run = np.random.default_rng(args.seed * 7 + L * 1000 + int(delta * 1e6))
                t0 = time.time()
                earliest, last_correct = study_inputs(
                    F_bits, grid=L,
                    c_pre=args.c_pre, c_amp=args.c_amp, c_shake=args.c_shake,
                    c_final=args.c_final, k_swap=args.k_swap, num_shakes=args.num_shakes,
                    initial_states=inits, labels=labs,
                    rng=rng_run, backend=backend,
                )
                # Aggregate phase histogram
                n = len(labs)
                phase_counts = {}
                for p in earliest:
                    phase_counts[p] = phase_counts.get(p, 0) + 1
                # 'final_amp_required' = trials where the final amp first reached consensus
                final_amp_needed = phase_counts.get("after_final_amp", 0)
                never_reached = phase_counts.get("never_reached", 0)
                early_consensus = n - final_amp_needed - never_reached
                row = {
                    "L": L, "delta": delta, "dataset": dataset_name,
                    "n_trials": n,
                    "phase_counts": phase_counts,
                    "early_consensus": early_consensus,
                    "final_amp_needed": final_amp_needed,
                    "never_reached": never_reached,
                    "correct": int(sum(last_correct)),
                    "elapsed_seconds": time.time() - t0,
                }
                runs.append(row)
                print(
                    f"L={L:>3d} d={delta:<5.3f} {dataset_name:<12} "
                    f"early={early_consensus:>3d}/{n} ({100*early_consensus/n:>4.1f}%)  "
                    f"final_amp_needed={final_amp_needed:>3d}  "
                    f"never={never_reached:>3d}  "
                    f"[{row['elapsed_seconds']:.1f}s]"
                )

    args.output.write_text(json.dumps({
        "schedule_params": {
            "sid": args.sid,
            "c_pre": args.c_pre, "c_amp": args.c_amp,
            "c_shake": args.c_shake, "c_final": args.c_final,
            "k_swap": args.k_swap, "num_shakes": args.num_shakes,
        },
        "runs": runs,
    }, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
