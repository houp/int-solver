"""Evaluate finite-switch schedules with various amplifier sequences.

Schedule form:   preprocessor F^{T_1}  then  amplifier-sequence^{T_2}

where the amplifier-sequence is a cyclic list of named local-majority rules
(see amplifier_library.py). Multiple amplifier schedules are compared.

Two diagnostic modes:

1. `--amplifier-only`: skip the preprocessor and evaluate each amplifier
   schedule from random initial configurations. Reveals which amplifiers
   have a trivial fixed-point problem and which do not.

2. normal mode: apply preprocessor F for T_1 steps (with F = catalog NCCA
   resolved by --sid), then run the amplifier schedule for T_2 steps.
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

from amplifier_library import (
    AMPLIFIERS,
    RADIUS_KIND_SPECS,
    apply_radius2_step,
    apply_radius_step_mlx,
    build_amplifier,
)

RADIUS2_KINDS = set(RADIUS_KIND_SPECS.keys())
from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend


def run_schedule(
    preprocessor_bits: np.ndarray | None,
    T1: int,
    amplifier_names: list[str],
    T2: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    backend,
) -> dict:
    batch, h, w = initial_states.shape
    L2 = h * w

    states = backend.asarray(initial_states, dtype="uint8")
    if preprocessor_bits is not None and T1 > 0:
        tiled_F = backend.asarray(np.tile(preprocessor_bits, (batch, 1)), dtype="uint8")
        for _ in range(T1):
            states = backend.step_pairwise(states, tiled_F)

    if T2 > 0 and amplifier_names:
        # Pre-build a tiled LUT per distinct LUT-type amplifier name
        tiled_luts: dict[str, np.ndarray] = {}
        for name in set(amplifier_names):
            if name in RADIUS2_KINDS:
                continue
            bits = np.asarray(build_amplifier(name), dtype=np.uint8)
            tiled_luts[name] = backend.asarray(
                np.tile(bits, (batch, 1)), dtype="uint8"
            )
        cycle_len = len(amplifier_names)
        using_mlx = backend.name == "mlx"
        for step in range(T2):
            name = amplifier_names[step % cycle_len]
            if name in RADIUS2_KINDS:
                if using_mlx:
                    states = apply_radius_step_mlx(states, name)
                else:
                    s_np = backend.to_numpy(states)
                    s_np = apply_radius2_step(s_np, name)
                    states = backend.asarray(s_np, dtype="uint8")
            else:
                states = backend.step_pairwise(states, tiled_luts[name])

    final = backend.to_numpy(states)
    totals = final.reshape(batch, -1).sum(axis=1)
    all_zero = (totals == 0)
    all_one = (totals == L2)
    consensus = all_zero | all_one
    correct = (all_zero & (labels == 0)) | (all_one & (labels == 1))
    final_majority = (totals > (L2 / 2)).astype(np.uint8)

    # Also report mean final density (how biased the non-consensus states are)
    final_density = float(final.mean())

    return {
        "consensus_rate": float(consensus.mean()),
        "correct_consensus_rate": float(correct.mean()),
        "final_majority_accuracy": float((final_majority == labels).mean()),
        "final_density_mean": final_density,
    }


def build_initial_states(rng, grid, probabilities, trials_per_side):
    inits = []
    labels = []
    prob_col = []
    for p in probabilities:
        init = (rng.random((trials_per_side, grid, grid)) < p).astype(np.uint8)
        totals = init.reshape(trials_per_side, -1).sum(axis=1)
        lab = (totals > (grid * grid / 2)).astype(np.uint8)
        inits.append(init)
        labels.append(lab)
        prob_col.extend([p] * trials_per_side)
    return np.concatenate(inits), np.concatenate(labels), prob_col


def parse_schedule_spec(spec: str) -> list[str]:
    """spec like 'row3,col3' or 'moore9' returns list of amplifier names."""
    return [s.strip() for s in spec.split(",") if s.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument(
        "--sid",
        default=None,
        help="Preprocessor NCCA sid. If omitted or --amplifier-only, no preprocessor is used.",
    )
    parser.add_argument("--T1", type=int, default=0)
    parser.add_argument("--T2", type=int, default=256)
    parser.add_argument(
        "--amplifier-schedules",
        nargs="+",
        required=True,
        help="Schedules like 'row3,col3' 'moore9' 'vn5,diag5' — each is a cyclic sequence",
    )
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--trials-per-side", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx",
                        help="Simulator backend: numpy or mlx (default mlx for Apple Silicon)")
    parser.add_argument("--amplifier-only", action="store_true",
                        help="Skip preprocessor entirely; evaluate amplifier from random init")
    parser.add_argument("--output", type=Path, default=Path("amplifier_test.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))

    F_bits: np.ndarray | None = None
    resolved_sid: str | None = None
    if args.sid and not args.amplifier_only:
        idx = catalog.resolve_rule_ref(args.sid)
        F_bits = catalog.lut_bits[idx].astype(np.uint8)
        resolved_sid = str(catalog.stable_ids[idx])

    backend = create_backend(args.backend)
    rng = np.random.default_rng(args.seed)

    schedules = [parse_schedule_spec(s) for s in args.amplifier_schedules]

    runs = []
    for grid in args.grids:
        initial_states, labels, prob_col = build_initial_states(
            rng, grid, args.probabilities, args.trials_per_side
        )
        for schedule_names, schedule_spec in zip(schedules, args.amplifier_schedules):
            t0 = time.time()
            per_prob = []
            for p in args.probabilities:
                mask = np.asarray([float(pp) == float(p) for pp in prob_col], dtype=bool)
                r = run_schedule(
                    F_bits,
                    args.T1 if not args.amplifier_only else 0,
                    schedule_names,
                    args.T2,
                    initial_states[mask],
                    labels[mask],
                    backend,
                )
                r["probability"] = float(p)
                per_prob.append(r)
            row = {
                "amplifier_schedule": schedule_spec,
                "T1": args.T1 if not args.amplifier_only else 0,
                "T2": args.T2,
                "grid": grid,
                "sid": resolved_sid,
                "amplifier_only": bool(args.amplifier_only),
                "per_probability": per_prob,
                "min_correct_consensus_rate": min(x["correct_consensus_rate"] for x in per_prob),
                "min_consensus_rate": min(x["consensus_rate"] for x in per_prob),
                "min_final_majority_accuracy": min(x["final_majority_accuracy"] for x in per_prob),
                "elapsed_seconds": time.time() - t0,
            }
            runs.append(row)
            print(
                f"grid={grid:3d} schedule={schedule_spec:<25} "
                f"min_cc={row['min_correct_consensus_rate']:.3f} "
                f"min_cons={row['min_consensus_rate']:.3f} "
                f"min_fma={row['min_final_majority_accuracy']:.3f}"
            )

    out = {
        "T1": args.T1 if not args.amplifier_only else 0,
        "T2": args.T2,
        "amplifier_only": args.amplifier_only,
        "preprocessor_sid": resolved_sid,
        "runs": runs,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
