"""Diagnose residual failures of a (preprocessor, amplifier) schedule.

Runs the schedule on many random trials, captures *per-trial*:
- initial state (for post hoc inspection)
- state after preprocessor F^T1
- state after amplifier schedule
- whether the trial succeeded (all-correct consensus) or failed
- if failed, whether the amplifier reached a fixed point, a period-k cycle,
  or is still evolving at termination

Outputs:
- a JSON summary with aggregate statistics
- an npz bundle with the raw initial / after-F / final states for failing
  and (a few) succeeding trials, so we can diff them

The purpose is to understand *what structural property of the initial state
or the after-F state* predicts failure, so we can design a better
preprocessor / amplifier combination.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


# --- repo-root path bootstrap (keep BEFORE ca_search/local imports) ---
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
_CATALOG_DIR = _REPO_ROOT / "catalogs"
# ---------------------------------------------------------------------

import numpy as np
from scipy import ndimage

from amplifier_library import (
    RADIUS_KIND_SPECS,
    apply_radius2_step,
    apply_radius_step_mlx,
    build_amplifier,
)
from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend

RADIUS_KINDS = set(RADIUS_KIND_SPECS.keys())


def run_schedule_with_snapshots(
    F_bits: np.ndarray,
    T1: int,
    amp_names: list[str],
    T2: int,
    initial_states: np.ndarray,
    *,
    backend,
    fixed_point_window: int,
):
    """Run the full schedule and return (after_F, final, cycle_info).

    cycle_info: per trial, (period, steps_in_cycle) where period is:
    - 1 = fixed point reached at least fixed_point_window steps before end
    - >1 = periodic cycle detected by comparing state to state `period` steps ago
    - 0 = still evolving at termination
    """
    batch = initial_states.shape[0]
    states = backend.asarray(initial_states, dtype="uint8")

    # preprocessor
    if F_bits is not None and T1 > 0:
        tiled_F = backend.asarray(np.tile(F_bits, (batch, 1)), dtype="uint8")
        for _ in range(T1):
            states = backend.step_pairwise(states, tiled_F)
    after_F = backend.to_numpy(states).copy()

    # amplifier
    tiled_luts: dict[str, np.ndarray] = {}
    using_mlx = backend.name == "mlx"
    for name in set(amp_names):
        if name in RADIUS_KINDS:
            continue
        bits = np.asarray(build_amplifier(name), dtype=np.uint8)
        tiled_luts[name] = backend.asarray(np.tile(bits, (batch, 1)), dtype="uint8")

    history: list[np.ndarray] = []
    cycle_len = len(amp_names)
    for step in range(T2):
        name = amp_names[step % cycle_len]
        if name in RADIUS_KINDS:
            if using_mlx:
                states = apply_radius_step_mlx(states, name)
            else:
                s_np = backend.to_numpy(states)
                s_np = apply_radius2_step(s_np, name)
                states = backend.asarray(s_np, dtype="uint8")
        else:
            states = backend.step_pairwise(states, tiled_luts[name])

        # Only keep the last fixed_point_window states for fixed-point / cycle detection
        if T2 - step <= fixed_point_window:
            history.append(backend.to_numpy(states).copy())

    final = history[-1]
    cycle_periods: list[int] = []
    for i in range(batch):
        period = 0
        for p in (1, 2, cycle_len, 2 * cycle_len, 3 * cycle_len, 4 * cycle_len):
            if len(history) < 2 * p:
                continue
            if np.array_equal(history[-1][i], history[-1 - p][i]):
                period = p
                break
        cycle_periods.append(period)

    return after_F, final, cycle_periods


def per_trial_features(state: np.ndarray) -> dict:
    h, w = state.shape
    density = float(state.mean())
    iface_h = int((state != np.roll(state, -1, axis=0)).sum())
    iface_v = int((state != np.roll(state, -1, axis=1)).sum())
    iface = iface_h + iface_v

    # Largest 4-connected component per colour
    def largest(val):
        mask = (state == val).astype(np.uint8)
        labeled, n = ndimage.label(mask)
        if n == 0:
            return 0, 0
        sizes = np.bincount(labeled.ravel())[1:]
        return int(n), int(sizes.max())

    n0, lg0 = largest(0)
    n1, lg1 = largest(1)

    # Stripe / uniform anisotropy: fraction of interface that is horizontal vs vertical
    horiz_frac = float(iface_h) / iface if iface > 0 else 0.0

    # Check for checkerboard-ish patterns (every nearest neighbor differs)
    # on the 2D lattice
    nbr_diff_right = int((state != np.roll(state, -1, axis=1)).sum())
    nbr_diff_down = int((state != np.roll(state, -1, axis=0)).sum())

    return {
        "density": density,
        "iface_total": iface,
        "iface_fraction_horizontal": horiz_frac,
        "components_zero": n0,
        "largest_zero": lg0,
        "components_one": n1,
        "largest_one": lg1,
        "largest_zero_frac": lg0 / float(h * w),
        "largest_one_frac": lg1 / float(h * w),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sid", default="sid:58ed6b657afb")
    parser.add_argument("--T1", type=int, default=128)
    parser.add_argument("--T2", type=int, default=2048)
    parser.add_argument("--amplifier", default="moore81")
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--trials-per-side", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fixed-point-window", type=int, default=32)
    parser.add_argument("--snapshots-path", type=Path, default=Path("failure_snapshots.npz"))
    parser.add_argument("--output", type=Path, default=Path("failure_diagnostic.json"))
    parser.add_argument("--backend", default="mlx")
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    idx = catalog.resolve_rule_ref(args.sid)
    F_bits = catalog.lut_bits[idx].astype(np.uint8)
    resolved_sid = str(catalog.stable_ids[idx])
    backend = create_backend(args.backend)
    amp_names = [s.strip() for s in args.amplifier.split(",") if s.strip()]

    rng = np.random.default_rng(args.seed)

    all_records = []
    init_bundles = []
    after_F_bundles = []
    final_bundles = []
    labels_bundles = []

    t0 = time.time()
    for p in args.probabilities:
        init = (rng.random((args.trials_per_side, args.grid, args.grid)) < p).astype(np.uint8)
        totals = init.reshape(args.trials_per_side, -1).sum(axis=1)
        labels = (totals > (args.grid * args.grid / 2)).astype(np.uint8)

        after_F, final, periods = run_schedule_with_snapshots(
            F_bits, args.T1, amp_names, args.T2, init,
            backend=backend, fixed_point_window=args.fixed_point_window,
        )

        final_totals = final.reshape(args.trials_per_side, -1).sum(axis=1)
        all_zero = (final_totals == 0)
        all_one = (final_totals == args.grid * args.grid)
        consensus = all_zero | all_one
        correct = (all_zero & (labels == 0)) | (all_one & (labels == 1))

        init_bundles.append(init)
        after_F_bundles.append(after_F)
        final_bundles.append(final)
        labels_bundles.append(labels)

        for i in range(args.trials_per_side):
            rec = {
                "probability": float(p),
                "trial": i,
                "initial_density": float(init[i].mean()),
                "initial_label": int(labels[i]),
                "tied_initial": bool(totals[i] * 2 == args.grid * args.grid),
                "final_density": float(final[i].mean()),
                "consensus": bool(consensus[i]),
                "correct": bool(correct[i]),
                "cycle_period": int(periods[i]),
                "after_F_features": per_trial_features(after_F[i]),
                "final_features": per_trial_features(final[i]),
            }
            all_records.append(rec)

    init_all = np.concatenate(init_bundles)
    after_F_all = np.concatenate(after_F_bundles)
    final_all = np.concatenate(final_bundles)
    labels_all = np.concatenate(labels_bundles)

    # Save snapshots (compressed)
    np.savez_compressed(
        args.snapshots_path,
        initial=init_all,
        after_F=after_F_all,
        final=final_all,
        labels=labels_all,
    )

    # Aggregate
    total = len(all_records)
    failures = [r for r in all_records if not r["correct"]]
    consensus_failures = [r for r in failures if r["consensus"]]  # reached wrong consensus
    nonconsensus_failures = [r for r in failures if not r["consensus"]]

    summary = {
        "sid": resolved_sid,
        "grid": args.grid,
        "T1": args.T1,
        "T2": args.T2,
        "amplifier": args.amplifier,
        "probabilities": args.probabilities,
        "trials_per_side": args.trials_per_side,
        "total_trials": total,
        "failures_total": len(failures),
        "consensus_failures": len(consensus_failures),
        "nonconsensus_failures": len(nonconsensus_failures),
        "per_trial": all_records,
        "snapshots_path": str(args.snapshots_path),
        "elapsed_seconds": time.time() - t0,
    }

    args.output.write_text(json.dumps(summary, indent=2))
    print(f"total={total} failures={len(failures)} "
          f"(consensus_failures={len(consensus_failures)}, "
          f"nonconsensus_failures={len(nonconsensus_failures)})")
    print(f"wrote {args.output} and {args.snapshots_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
