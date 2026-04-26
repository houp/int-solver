"""Universal scaled schedule test.

Schedule with time-scaling proportional to L:
    T_pre        = c_pre * L     (default c_pre = 0.5)
    T_amp        = c_amp * L     (default c_amp = 1.0)
    T_shake      = c_shake * L   (default c_shake = 0.25)
    T_amp_final  = c_final * L   (default c_final = 4.0)
    K_swaps      = k_swap * L    (default k_swap = 8)

Tests on both random and structured adversarial inputs at multiple grids.
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
from conservative_noise import apply_lut_mlx, apply_swaps
from adversarial_realistic import build_adversarial_inits


def run_schedule(
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
        "schedule": {"T_pre": T_pre, "T_amp": T_amp, "T_shake": T_shake,
                      "T_amp_final": T_amp_final, "K_swaps_per_shake": K, "num_shakes": num_shakes},
        "consensus_rate": float(consensus.mean()),
        "correct_consensus_rate": float(correct.mean()),
        "num_failures": int((~correct).sum()),
        "per_trial_correct": correct.tolist(),
    }


def build_random_inits(rng, grid, probabilities, trials_per_side):
    inits, labs = [], []
    for p in probabilities:
        init = (rng.random((trials_per_side, grid, grid)) < p).astype(np.uint8)
        totals = init.reshape(trials_per_side, -1).sum(axis=1)
        lab = (totals > (grid * grid / 2)).astype(np.uint8)
        inits.append(init); labs.append(lab)
    return np.concatenate(inits), np.concatenate(labs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sid", default="sid:58ed6b657afb")
    parser.add_argument("--c-pre", type=float, default=0.5)
    parser.add_argument("--c-amp", type=float, default=1.0)
    parser.add_argument("--c-shake", type=float, default=0.25)
    parser.add_argument("--c-final", type=float, default=4.0)
    parser.add_argument("--k-swap", type=float, default=8.0)
    parser.add_argument("--num-shakes", type=int, default=2)
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 128, 192, 256, 384, 512])
    parser.add_argument("--trials-per-side-random", type=int, default=64)
    parser.add_argument("--random-probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--adv-densities", type=float, nargs="+", default=[0.51, 0.52, 0.55])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("scaled_schedule.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    F_bits = catalog.lut_bits[catalog.resolve_rule_ref(args.sid)].astype(np.uint8)
    backend = create_backend(args.backend)

    all_runs = []
    for grid in args.grids:
        print(f"=== grid={grid} ===")
        # Random
        rng_rand = np.random.default_rng(args.seed + grid)
        random_inits, random_labels = build_random_inits(
            rng_rand, grid, args.random_probabilities, args.trials_per_side_random)
        rng_stream = np.random.default_rng(args.seed * 31 + grid)
        t0 = time.time()
        r = run_schedule(
            F_bits, grid=grid,
            c_pre=args.c_pre, c_amp=args.c_amp, c_shake=args.c_shake,
            c_final=args.c_final, k_swap=args.k_swap, num_shakes=args.num_shakes,
            initial_states=random_inits, labels=random_labels,
            rng=rng_stream, backend=backend,
        )
        r["grid"] = grid; r["dataset"] = "random"; r["trials"] = len(random_labels)
        r["elapsed_seconds"] = time.time() - t0
        all_runs.append(r)
        print(f"  random:      cc={r['correct_consensus_rate']:.4f} "
              f"fails={r['num_failures']}/{len(random_labels)} [{r['elapsed_seconds']:.1f}s] "
              f"sched={r['schedule']}")

        # Adversarial
        rng_adv = np.random.default_rng(args.seed + grid + 99999)
        adv_inits, adv_labels, adv_names = build_adversarial_inits(
            grid, args.adv_densities, rng_adv)
        rng_stream = np.random.default_rng(args.seed * 37 + grid)
        t0 = time.time()
        r = run_schedule(
            F_bits, grid=grid,
            c_pre=args.c_pre, c_amp=args.c_amp, c_shake=args.c_shake,
            c_final=args.c_final, k_swap=args.k_swap, num_shakes=args.num_shakes,
            initial_states=adv_inits, labels=adv_labels,
            rng=rng_stream, backend=backend,
        )
        r["grid"] = grid; r["dataset"] = "adversarial"; r["trials"] = len(adv_labels)
        r["elapsed_seconds"] = time.time() - t0
        r["adv_case_names"] = adv_names
        all_runs.append(r)
        print(f"  adversarial: cc={r['correct_consensus_rate']:.4f} "
              f"fails={r['num_failures']}/{len(adv_labels)} [{r['elapsed_seconds']:.1f}s]")
        correct_arr = np.asarray(r["per_trial_correct"])
        if r["num_failures"] > 0:
            for i, name in enumerate(adv_names):
                if not correct_arr[i]:
                    print(f"    FAIL: {name}  label={adv_labels[i]}")

    args.output.write_text(json.dumps({
        "sid": args.sid, "runs": all_runs,
        "params": {"c_pre": args.c_pre, "c_amp": args.c_amp,
                   "c_shake": args.c_shake, "c_final": args.c_final,
                   "k_swap": args.k_swap, "num_shakes": args.num_shakes},
    }, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
