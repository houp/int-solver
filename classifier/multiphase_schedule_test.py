"""Test multi-phase finite-switch schedules.

Given a sequence of rule sids and corresponding phase lengths, evaluates:

    F_1^{T_1} F_2^{T_2} ... F_k^{T_k}  then  majority^{T_m}

(Each F_i run once for T_i steps, no outer repetition; single switch to
majority at the end.)

Use case: test whether a two-stage NCCA preprocessor (e.g. first a strong
coarsener, then a disperser) followed by majority can reach correct
consensus where single-rule schedules cannot.
"""
from __future__ import annotations

import argparse
import itertools
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


def run_multiphase(
    catalog,
    phase_sids: list[str],
    phase_lengths: list[int],
    amplifier_name: str,
    amplifier_steps: int,
    *,
    grid: int,
    probabilities: list[float],
    trials_per_side: int,
    seed: int,
    backend_name: str,
) -> dict:
    if amplifier_name == "moore":
        G = np.asarray(moore_majority_rule_bits(), dtype=np.uint8)
    else:
        G = np.asarray(von_neumann_majority_rule_bits(), dtype=np.uint8)

    backend = create_backend(backend_name)
    rng = np.random.default_rng(seed)

    rule_bits = [
        catalog.lut_bits[catalog.resolve_rule_ref(sid)].astype(np.uint8)
        for sid in phase_sids
    ]

    per_prob = []
    for p in probabilities:
        init = (rng.random((trials_per_side, grid, grid)) < p).astype(np.uint8)
        totals = init.reshape(trials_per_side, -1).sum(axis=1)
        labels = (totals > (grid * grid / 2)).astype(np.uint8)

        states = backend.asarray(init, dtype="uint8")
        for bits, T in zip(rule_bits, phase_lengths):
            tiled = backend.asarray(np.tile(bits, (trials_per_side, 1)), dtype="uint8")
            for _ in range(T):
                states = backend.step_pairwise(states, tiled)
        tiled_G = backend.asarray(np.tile(G, (trials_per_side, 1)), dtype="uint8")
        for _ in range(amplifier_steps):
            states = backend.step_pairwise(states, tiled_G)

        final = backend.to_numpy(states)
        final_totals = final.reshape(trials_per_side, -1).sum(axis=1)
        all_zero = (final_totals == 0)
        all_one = (final_totals == grid * grid)
        consensus = all_zero | all_one
        correct = (all_zero & (labels == 0)) | (all_one & (labels == 1))
        final_majority = (final_totals > (grid * grid / 2)).astype(np.uint8)
        per_prob.append({
            "probability": float(p),
            "trials": trials_per_side,
            "consensus_rate": float(consensus.mean()),
            "correct_consensus_rate": float(correct.mean()),
            "final_majority_accuracy": float((final_majority == labels).mean()),
        })

    return {
        "phase_sids": phase_sids,
        "phase_lengths": phase_lengths,
        "amplifier": amplifier_name,
        "amplifier_steps": amplifier_steps,
        "grid": grid,
        "per_probability": per_prob,
        "min_correct_consensus_rate": min(x["correct_consensus_rate"] for x in per_prob),
        "min_consensus_rate": min(x["consensus_rate"] for x in per_prob),
        "min_final_majority_accuracy": min(x["final_majority_accuracy"] for x in per_prob),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sids", nargs="+", required=True)
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--trials-per-side", type=int, default=16)
    parser.add_argument(
        "--phase-lengths",
        type=int,
        nargs="+",
        default=[32, 32],
        help="Phase length per rule; must have same length as --sids-per-schedule",
    )
    parser.add_argument(
        "--amplifier-steps",
        type=int,
        default=512,
        help="Number of majority steps at the end",
    )
    parser.add_argument("--amplifier", choices=["moore", "vn"], default="vn")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("multiphase.json"))
    parser.add_argument(
        "--pair-search",
        action="store_true",
        help="Enumerate ordered pairs from --sids rather than using them as a fixed sequence",
    )
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))

    runs = []
    if args.pair_search:
        pairs = [p for p in itertools.product(args.sids, args.sids) if p[0] != p[1]]
        total = len(pairs) * len(args.grids)
        done = 0
        t0 = time.time()
        for sid1, sid2 in pairs:
            for grid in args.grids:
                r = run_multiphase(
                    catalog,
                    [sid1, sid2],
                    args.phase_lengths,
                    args.amplifier,
                    args.amplifier_steps,
                    grid=grid,
                    probabilities=args.probabilities,
                    trials_per_side=args.trials_per_side,
                    seed=args.seed,
                    backend_name=args.backend,
                )
                runs.append(r)
                done += 1
                if done % max(1, total // 20) == 0:
                    print(f"  {done}/{total} ({time.time()-t0:.1f}s)")
        ranked = sorted(runs, key=lambda r: -r["min_correct_consensus_rate"])
        print()
        print(f"=== Top 20 pairs ===")
        for r in ranked[:20]:
            print(
                f"{r['phase_sids'][0][:16]:<16} + {r['phase_sids'][1][:16]:<16} "
                f"lengths={r['phase_lengths']} grid={r['grid']:3d} "
                f"min_cc={r['min_correct_consensus_rate']:.3f} "
                f"min_fma={r['min_final_majority_accuracy']:.3f}"
            )
    else:
        for grid in args.grids:
            r = run_multiphase(
                catalog,
                args.sids,
                args.phase_lengths,
                args.amplifier,
                args.amplifier_steps,
                grid=grid,
                probabilities=args.probabilities,
                trials_per_side=args.trials_per_side,
                seed=args.seed,
                backend_name=args.backend,
            )
            runs.append(r)
            print(
                f"sids={r['phase_sids']} lengths={r['phase_lengths']} "
                f"grid={grid} min_cc={r['min_correct_consensus_rate']:.3f} "
                f"min_fma={r['min_final_majority_accuracy']:.3f}"
            )

    args.output.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
