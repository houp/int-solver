"""Test whether coarsener NCCAs, followed by a single global switch to
local majority, actually solve the 2D density classification task.

For each given rule F, we run F for T1 steps, then plain Moore-majority for
T2 steps, and measure:

- consensus_rate: fraction of trials ending in all-0 or all-1
- consensus_accuracy: fraction ending in all-<true global majority>
- final_majority_accuracy: fraction whose final global majority matches initial

This is the pure Fukś-style finite-switch schedule (no periodic repetition).
We sweep T1 across several values to probe scaling with lattice size.
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

from ca_search.binary_catalog import load_binary_catalog
from ca_search.density_classification import (
    moore_majority_rule_bits,
    von_neumann_majority_rule_bits,
)
from ca_search.simulator import create_backend


def run_finite_switch(
    catalog,
    sid: str,
    *,
    grid: int,
    probabilities: list[float],
    trials_per_side: int,
    T1: int,
    T2: int,
    amplifier_name: str,
    seed: int,
    backend_name: str,
) -> dict:
    backend = create_backend(backend_name)
    rng = np.random.default_rng(seed)
    idx = catalog.resolve_rule_ref(sid)
    F = catalog.lut_bits[idx].astype(np.uint8)
    if amplifier_name == "moore":
        G = np.asarray(moore_majority_rule_bits(), dtype=np.uint8)
    elif amplifier_name == "vn":
        G = np.asarray(von_neumann_majority_rule_bits(), dtype=np.uint8)
    else:
        raise ValueError(f"unknown amplifier: {amplifier_name}")

    per_prob = []
    for p in probabilities:
        init = (rng.random((trials_per_side, grid, grid)) < p).astype(np.uint8)
        totals = init.reshape(trials_per_side, -1).sum(axis=1)
        labels = (totals > (grid * grid / 2)).astype(np.uint8)
        tied = (totals * 2 == grid * grid)

        states = backend.asarray(init, dtype="uint8")
        tiled_F = backend.asarray(np.tile(F, (trials_per_side, 1)), dtype="uint8")
        tiled_G = backend.asarray(np.tile(G, (trials_per_side, 1)), dtype="uint8")

        for _ in range(T1):
            states = backend.step_pairwise(states, tiled_F)
        after_F = backend.to_numpy(states)
        for _ in range(T2):
            states = backend.step_pairwise(states, tiled_G)
        final = backend.to_numpy(states)

        final_totals = final.reshape(trials_per_side, -1).sum(axis=1)
        final_majority = (final_totals > (grid * grid / 2)).astype(np.uint8)
        all_zero = (final_totals == 0)
        all_one = (final_totals == grid * grid)
        consensus_rate = float((all_zero | all_one).mean())

        # Correct consensus: lattice fully 0 matches label 0; fully 1 matches label 1
        correct_consensus = (
            (all_zero & (labels == 0)) | (all_one & (labels == 1))
        )
        # Only over trials that achieved consensus
        cons_count = int((all_zero | all_one).sum())
        consensus_accuracy = (
            float(correct_consensus.sum()) / cons_count if cons_count > 0 else float("nan")
        )
        # Fraction of trials ending in correct consensus (over all trials)
        correct_consensus_rate = float(correct_consensus.mean())
        final_majority_accuracy = float(
            (final_majority == labels).mean()
        )

        # Intermediate (after F) quick stats
        iface_h = (after_F != np.roll(after_F, -1, axis=1)).sum(axis=(1, 2))
        iface_v = (after_F != np.roll(after_F, -1, axis=2)).sum(axis=(1, 2))
        iface_frac = float(((iface_h + iface_v) / (grid * grid)).mean())

        per_prob.append(
            {
                "probability": float(p),
                "trials": int(trials_per_side),
                "tied_initial_states": int(tied.sum()),
                "consensus_rate": consensus_rate,
                "correct_consensus_rate": correct_consensus_rate,
                "consensus_accuracy": consensus_accuracy,
                "final_majority_accuracy": final_majority_accuracy,
                "after_F_interface_fraction": iface_frac,
            }
        )

    return {
        "sid": sid,
        "grid": grid,
        "T1": T1,
        "T2": T2,
        "amplifier": amplifier_name,
        "per_probability": per_prob,
        "min_correct_consensus_rate": min(x["correct_consensus_rate"] for x in per_prob),
        "min_final_majority_accuracy": min(x["final_majority_accuracy"] for x in per_prob),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sids", nargs="+", required=True)
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--trials-per-side", type=int, default=16)
    parser.add_argument("--T1-schedule",
        choices=["fixed", "L", "LlogL", "L2"],
        default="L",
        help="How T1 scales with grid size L")
    parser.add_argument("--T1-fixed", type=int, default=128)
    parser.add_argument("--T2", type=int, default=128)
    parser.add_argument("--amplifier", choices=["moore", "vn"], default="vn")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("finite_switch_coarsener.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))

    runs = []
    for sid in args.sids:
        for grid in args.grids:
            if args.T1_schedule == "fixed":
                T1 = args.T1_fixed
            elif args.T1_schedule == "L":
                T1 = grid
            elif args.T1_schedule == "LlogL":
                T1 = int(grid * np.log2(grid))
            elif args.T1_schedule == "L2":
                T1 = grid * grid
            t0 = time.time()
            r = run_finite_switch(
                catalog,
                sid,
                grid=grid,
                probabilities=args.probabilities,
                trials_per_side=args.trials_per_side,
                T1=T1,
                T2=args.T2,
                amplifier_name=args.amplifier,
                seed=args.seed,
                backend_name=args.backend,
            )
            r["elapsed_seconds"] = time.time() - t0
            print(
                f"sid={sid[:18]:<18} grid={grid:>4d} T1={T1:>5d} "
                f"min_correct_consensus={r['min_correct_consensus_rate']:.3f} "
                f"min_final_majority_accuracy={r['min_final_majority_accuracy']:.3f} "
                f"iface_frac_after_F={[x['after_F_interface_fraction'] for x in r['per_probability']]}"
            )
            runs.append(r)

    args.output.write_text(
        json.dumps(
            {
                "T1_schedule": args.T1_schedule,
                "T1_fixed": args.T1_fixed,
                "T2": args.T2,
                "amplifier": args.amplifier,
                "trials_per_side": args.trials_per_side,
                "probabilities": args.probabilities,
                "runs": runs,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
