"""Heavy validation of the current best schedule at multiple grid sizes and
many random seeds. Confirms (or refutes) the 100% claim.

Uses the schedule:
    sid:58ed6^{T_pre}  [moore81^{T_amp}  F_shake^{T_shake}]^{num_shakes}  moore81^{T_amp_final}
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

from amplifier_library import apply_radius_step_mlx
from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend


def run_trial_batch(
    catalog,
    sid_pre: str,
    sid_shake: str,
    *,
    T_pre: int,
    T_amp: int,
    T_shake: int,
    num_shakes: int,
    T_amp_final: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    backend,
) -> dict:
    idx_pre = catalog.resolve_rule_ref(sid_pre)
    F_pre = catalog.lut_bits[idx_pre].astype(np.uint8)
    idx_shake = catalog.resolve_rule_ref(sid_shake)
    F_shake = catalog.lut_bits[idx_shake].astype(np.uint8)

    batch, h, w = initial_states.shape
    L2 = h * w
    states = backend.asarray(initial_states, dtype="uint8")
    tiled_pre = backend.asarray(np.tile(F_pre, (batch, 1)), dtype="uint8")
    tiled_shake = backend.asarray(np.tile(F_shake, (batch, 1)), dtype="uint8")

    for _ in range(T_pre):
        states = backend.step_pairwise(states, tiled_pre)
    for _ in range(num_shakes):
        for _ in range(T_amp):
            states = apply_radius_step_mlx(states, "moore81")
        for _ in range(T_shake):
            states = backend.step_pairwise(states, tiled_shake)
    for _ in range(T_amp_final):
        states = apply_radius_step_mlx(states, "moore81")

    final = backend.to_numpy(states)
    totals = final.reshape(batch, -1).sum(axis=1)
    all_zero = (totals == 0)
    all_one = (totals == L2)
    correct = (all_zero & (labels == 0)) | (all_one & (labels == 1))
    final_majority = (totals > (L2 / 2)).astype(np.uint8)

    return {
        "trials": int(batch),
        "consensus_rate": float((all_zero | all_one).mean()),
        "correct_consensus_rate": float(correct.mean()),
        "final_majority_accuracy": float((final_majority == labels).mean()),
        "num_failures": int((~correct).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sid-pre", default="sid:58ed6b657afb")
    parser.add_argument("--sid-shake", default="sid:58ed6b657afb")
    parser.add_argument("--grids", type=int, nargs="+", default=[192, 256, 384])
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--trials-per-seed", type=int, default=32,
                        help="Trials per probability per seed")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[2026, 4127, 6228, 8329, 10430])
    parser.add_argument("--T-pre", type=int, default=128)
    parser.add_argument("--T-amp", type=int, default=256)
    parser.add_argument("--T-shake", type=int, default=64)
    parser.add_argument("--num-shakes", type=int, default=2)
    parser.add_argument("--T-amp-final", type=int, default=1024)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("validation_big.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    backend = create_backend(args.backend)

    runs = []
    for grid in args.grids:
        total_trials = 0
        total_failures = 0
        per_seed_rows = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            t0 = time.time()
            inits, labs = [], []
            for p in args.probabilities:
                init = (rng.random((args.trials_per_seed, grid, grid)) < p).astype(np.uint8)
                totals = init.reshape(args.trials_per_seed, -1).sum(axis=1)
                lab = (totals > (grid * grid / 2)).astype(np.uint8)
                inits.append(init)
                labs.append(lab)
            initial_states = np.concatenate(inits)
            labels = np.concatenate(labs)

            result = run_trial_batch(
                catalog, args.sid_pre, args.sid_shake,
                T_pre=args.T_pre,
                T_amp=args.T_amp,
                T_shake=args.T_shake,
                num_shakes=args.num_shakes,
                T_amp_final=args.T_amp_final,
                initial_states=initial_states,
                labels=labels,
                backend=backend,
            )
            result["seed"] = int(seed)
            result["grid"] = grid
            result["elapsed_seconds"] = time.time() - t0
            per_seed_rows.append(result)
            total_trials += result["trials"]
            total_failures += result["num_failures"]
            print(
                f"grid={grid:>4d} seed={seed:>6d} "
                f"trials={result['trials']:>3d} "
                f"failures={result['num_failures']:>2d} "
                f"cc_rate={result['correct_consensus_rate']:.4f} "
                f"fma={result['final_majority_accuracy']:.4f} "
                f"[{result['elapsed_seconds']:.1f}s]"
            )

        aggregate = {
            "grid": grid,
            "total_trials": total_trials,
            "total_failures": total_failures,
            "aggregate_correct_consensus_rate": (total_trials - total_failures) / total_trials,
            "per_seed": per_seed_rows,
        }
        runs.append(aggregate)
        print(
            f"  *** grid={grid:>4d}  TOTAL: {total_trials - total_failures}/{total_trials} "
            f"= {aggregate['aggregate_correct_consensus_rate']:.4f} ***"
        )

    args.output.write_text(json.dumps({
        "sid_pre": args.sid_pre,
        "sid_shake": args.sid_shake,
        "amplifier": "moore81",
        "schedule": {
            "T_pre": args.T_pre,
            "T_amp": args.T_amp,
            "T_shake": args.T_shake,
            "num_shakes": args.num_shakes,
            "T_amp_final": args.T_amp_final,
        },
        "probabilities": args.probabilities,
        "trials_per_seed": args.trials_per_seed,
        "seeds": args.seeds,
        "runs": runs,
    }, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
