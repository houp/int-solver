"""Finite-switch schedule with a separate shake NCCA.

    F_pre^{T_pre}  [amp^{T_amp} F_shake^{T_shake}]^{num_shakes}  amp^{T_amp_final}

F_pre is the main preprocessor (e.g. sid:58ed6 outer_monotone rule).
F_shake is a different NCCA used only inside shakes (e.g. a dispersing
fragmenter). The shake rule's job is to break large minority blobs that
the large-radius amplifier cannot erode within finite T_amp.

The goal is to increase the correct-consensus rate beyond what the baseline
single-preprocessor finite-switch achieves, by targeting the specific
failure mode (large low-curvature blobs) identified by failure_diagnostic.
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
    RADIUS_KIND_SPECS,
    apply_radius_step_mlx,
    apply_radius2_step,
    build_amplifier,
)
from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend


RADIUS_KINDS = set(RADIUS_KIND_SPECS.keys())


def run_amp_phase(states, amp_names, steps, backend, tiled_luts, using_mlx):
    cycle = len(amp_names)
    for step in range(steps):
        name = amp_names[step % cycle]
        if name in RADIUS_KINDS:
            if using_mlx:
                states = apply_radius_step_mlx(states, name)
            else:
                s_np = backend.to_numpy(states)
                s_np = apply_radius2_step(s_np, name)
                states = backend.asarray(s_np, dtype="uint8")
        else:
            states = backend.step_pairwise(states, tiled_luts[name])
    return states


def run_schedule(
    F_pre_bits: np.ndarray,
    F_shake_bits: np.ndarray,
    amp_names: list[str],
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
    batch, h, w = initial_states.shape
    L2 = h * w
    using_mlx = backend.name == "mlx"
    states = backend.asarray(initial_states, dtype="uint8")
    tiled_pre = backend.asarray(np.tile(F_pre_bits, (batch, 1)), dtype="uint8")
    tiled_shake = backend.asarray(np.tile(F_shake_bits, (batch, 1)), dtype="uint8")
    tiled_luts: dict[str, np.ndarray] = {}
    for name in set(amp_names):
        if name in RADIUS_KINDS:
            continue
        bits = np.asarray(build_amplifier(name), dtype=np.uint8)
        tiled_luts[name] = backend.asarray(np.tile(bits, (batch, 1)), dtype="uint8")

    # Preprocessor phase
    for _ in range(T_pre):
        states = backend.step_pairwise(states, tiled_pre)

    # Amplifier + shake cycles
    for i in range(num_shakes):
        states = run_amp_phase(states, amp_names, T_amp, backend, tiled_luts, using_mlx)
        for _ in range(T_shake):
            states = backend.step_pairwise(states, tiled_shake)
    states = run_amp_phase(states, amp_names, T_amp_final, backend, tiled_luts, using_mlx)

    final = backend.to_numpy(states)
    totals = final.reshape(batch, -1).sum(axis=1)
    all_zero = (totals == 0)
    all_one = (totals == L2)
    consensus = all_zero | all_one
    correct = (all_zero & (labels == 0)) | (all_one & (labels == 1))
    final_majority = (totals > (L2 / 2)).astype(np.uint8)

    return {
        "consensus_rate": float(consensus.mean()),
        "correct_consensus_rate": float(correct.mean()),
        "final_majority_accuracy": float((final_majority == labels).mean()),
    }


def build_initial_states(rng, grid, probabilities, trials_per_side):
    inits, labs, prob_col = [], [], []
    for p in probabilities:
        init = (rng.random((trials_per_side, grid, grid)) < p).astype(np.uint8)
        totals = init.reshape(trials_per_side, -1).sum(axis=1)
        lab = (totals > (grid * grid / 2)).astype(np.uint8)
        inits.append(init)
        labs.append(lab)
        prob_col.extend([p] * trials_per_side)
    return np.concatenate(inits), np.concatenate(labs), prob_col


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sid-pre", default="sid:58ed6b657afb")
    parser.add_argument("--sid-shakes", nargs="+", default=["sid:58ed6b657afb"],
                        help="One or more shake rules to sweep")
    parser.add_argument("--amplifier", default="moore81")
    parser.add_argument("--T-pre", type=int, default=128)
    parser.add_argument("--T-amp", type=int, default=256)
    parser.add_argument("--T-shake-values", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--num-shakes-values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--T-amp-final", type=int, default=512)
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--trials-per-side", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("multi_switch_two_ncca.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    pre_idx = catalog.resolve_rule_ref(args.sid_pre)
    F_pre = catalog.lut_bits[pre_idx].astype(np.uint8)
    amp_names = [s.strip() for s in args.amplifier.split(",") if s.strip()]
    backend = create_backend(args.backend)

    rng = np.random.default_rng(args.seed)
    initial_states, labels, prob_col = build_initial_states(
        rng, args.grid, args.probabilities, args.trials_per_side
    )

    runs = []
    for shake_sid in args.sid_shakes:
        shake_idx = catalog.resolve_rule_ref(shake_sid)
        F_shake = catalog.lut_bits[shake_idx].astype(np.uint8)
        for T_shake in args.T_shake_values:
            for num_shakes in args.num_shakes_values:
                t0 = time.time()
                per_prob = []
                for p in args.probabilities:
                    mask = np.asarray([float(pp) == float(p) for pp in prob_col], dtype=bool)
                    r = run_schedule(
                        F_pre, F_shake, amp_names,
                        T_pre=args.T_pre,
                        T_amp=args.T_amp,
                        T_shake=T_shake,
                        num_shakes=num_shakes,
                        T_amp_final=args.T_amp_final,
                        initial_states=initial_states[mask],
                        labels=labels[mask],
                        backend=backend,
                    )
                    r["probability"] = float(p)
                    per_prob.append(r)
                summary = {
                    "shake_sid": shake_sid,
                    "T_pre": args.T_pre,
                    "T_amp": args.T_amp,
                    "T_shake": T_shake,
                    "num_shakes": num_shakes,
                    "T_amp_final": args.T_amp_final,
                    "grid": args.grid,
                    "amplifier": args.amplifier,
                    "per_probability": per_prob,
                    "min_correct_consensus_rate": min(x["correct_consensus_rate"] for x in per_prob),
                    "min_final_majority_accuracy": min(x["final_majority_accuracy"] for x in per_prob),
                    "elapsed_seconds": time.time() - t0,
                }
                runs.append(summary)
                print(
                    f"shake={shake_sid[:16]:<16} T_shake={T_shake:>3d} n={num_shakes} "
                    f"min_cc={summary['min_correct_consensus_rate']:.3f} "
                    f"min_fma={summary['min_final_majority_accuracy']:.3f} "
                    f"[{summary['elapsed_seconds']:.1f}s]"
                )

    args.output.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
