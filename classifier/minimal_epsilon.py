"""Q2: Minimal-stochasticity screen.

Take our best deterministic schedule:
    F^T_pre  [ moore81^T_amp  F_shake^T_shake ]^{num_shakes}  moore81^T_amp_final

and replace the F_shake phase with a *noisy* F_shake that applies F then
independently flips each cell with probability epsilon. This is the minimal
symmetry-breaking perturbation.

We sweep epsilon over several decades and measure correct-consensus rate on
both random inputs (baseline) and adversarial inputs (stripes/checker/etc.).
The question: does a small epsilon close the adversarial gap without hurting
the random-input 100%?
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


def _pair_indices_mlx(states):
    import mlx.core as mx
    x = mx.roll(mx.roll(states, 1, axis=1), 1, axis=2)
    y = mx.roll(states, 1, axis=1)
    z = mx.roll(mx.roll(states, 1, axis=1), -1, axis=2)
    t = mx.roll(states, 1, axis=2)
    u = states
    w = mx.roll(states, -1, axis=2)
    a = mx.roll(mx.roll(states, -1, axis=1), 1, axis=2)
    b = mx.roll(states, -1, axis=1)
    c = mx.roll(mx.roll(states, -1, axis=1), -1, axis=2)
    return (
        (x.astype(mx.uint16) << 8)
        | (y.astype(mx.uint16) << 7)
        | (z.astype(mx.uint16) << 6)
        | (t.astype(mx.uint16) << 5)
        | (u.astype(mx.uint16) << 4)
        | (w.astype(mx.uint16) << 3)
        | (a.astype(mx.uint16) << 2)
        | (b.astype(mx.uint16) << 1)
        | c.astype(mx.uint16)
    )


def apply_lut_mlx(states, tiled_bits):
    import mlx.core as mx
    indices = _pair_indices_mlx(states)
    out = mx.take_along_axis(tiled_bits[:, None, None, :], indices[..., None], axis=3)
    return mx.squeeze(out, axis=3).astype(mx.uint8)


def apply_noisy_F_mlx(states, tiled_bits, epsilon, rng):
    import mlx.core as mx
    F_out = apply_lut_mlx(states, tiled_bits)
    if epsilon <= 0:
        return F_out
    batch, H, W = states.shape
    mask_np = (rng.random((batch, H, W)) < epsilon).astype(np.uint8)
    if mask_np.sum() == 0:
        return F_out
    mask = mx.array(mask_np)
    return (F_out ^ mask).astype(mx.uint8)


def run_schedule_noisy(
    F_bits: np.ndarray,
    *,
    T_pre: int,
    T_amp: int,
    T_shake: int,
    num_shakes: int,
    T_amp_final: int,
    epsilon: float,
    initial_states: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    backend,
) -> dict:
    batch, H, W = initial_states.shape
    L2 = H * W
    states = backend.asarray(initial_states, dtype="uint8")
    tiled_F = backend.asarray(np.tile(F_bits, (batch, 1)), dtype="uint8")

    # Pre-phase: deterministic F
    for _ in range(T_pre):
        states = apply_lut_mlx(states, tiled_F)

    # Shake/amp cycles
    for _ in range(num_shakes):
        for _ in range(T_amp):
            states = apply_radius_step_mlx(states, "moore81")
        for _ in range(T_shake):
            states = apply_noisy_F_mlx(states, tiled_F, epsilon, rng)

    # Final amplification (deterministic)
    for _ in range(T_amp_final):
        states = apply_radius_step_mlx(states, "moore81")

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
        "num_failures": int((~correct).sum()),
    }


def build_adversarial_inits(grid: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ii, jj = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    cases, names, labels = [], [], []

    def with_tip(s, target):
        if target == 1:
            idx = np.argwhere(s == 0)
            if len(idx) > 0:
                s[idx[0, 0], idx[0, 1]] = 1
        else:
            idx = np.argwhere(s == 1)
            if len(idx) > 0:
                s[idx[0, 0], idx[0, 1]] = 0
        return s

    for target in (0, 1):
        sh = np.zeros((grid, grid), dtype=np.uint8); sh[::2] = 1
        cases.append(with_tip(sh, target).copy()); names.append(f"stripes_h_lbl{target}"); labels.append(target)
        sv = np.zeros((grid, grid), dtype=np.uint8); sv[:, ::2] = 1
        cases.append(with_tip(sv, target).copy()); names.append(f"stripes_v_lbl{target}"); labels.append(target)
        cb = ((ii + jj) % 2).astype(np.uint8)
        cases.append(with_tip(cb, target).copy()); names.append(f"checker_lbl{target}"); labels.append(target)
        bc = ((ii // 2 + jj // 2) % 2).astype(np.uint8)
        cases.append(with_tip(bc, target).copy()); names.append(f"block_checker_lbl{target}"); labels.append(target)
        hh = np.full((grid, grid), target, dtype=np.uint8); hh[:, : grid // 2] = 1 - target
        cases.append(with_tip(hh, target).copy()); names.append(f"half_half_lbl{target}"); labels.append(target)
    return np.stack(cases), np.asarray(labels, dtype=np.uint8), names


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
    parser.add_argument("--T-pre", type=int, default=128)
    parser.add_argument("--T-amp", type=int, default=256)
    parser.add_argument("--T-shake", type=int, default=64)
    parser.add_argument("--num-shakes", type=int, default=2)
    parser.add_argument("--T-amp-final", type=int, default=1024)
    parser.add_argument("--epsilon-values", type=float, nargs="+",
                        default=[0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 5e-1])
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--trials-per-side", type=int, default=128)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("minimal_epsilon.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    F_bits = catalog.lut_bits[catalog.resolve_rule_ref(args.sid)].astype(np.uint8)
    backend = create_backend(args.backend)

    runs = []
    for grid in args.grids:
        rng_init = np.random.default_rng(args.seed + grid)
        random_inits, random_labels = build_random_inits(rng_init, grid, args.probabilities, args.trials_per_side)
        adv_inits, adv_labels, adv_names = build_adversarial_inits(grid)

        for eps in args.epsilon_values:
            for dataset_name, inits, labs in [("random", random_inits, random_labels),
                                                ("adversarial", adv_inits, adv_labels)]:
                rng_stream = np.random.default_rng(args.seed + grid + int(eps * 1e9) + len(labs))
                t0 = time.time()
                r = run_schedule_noisy(
                    F_bits,
                    T_pre=args.T_pre, T_amp=args.T_amp,
                    T_shake=args.T_shake, num_shakes=args.num_shakes,
                    T_amp_final=args.T_amp_final,
                    epsilon=eps,
                    initial_states=inits, labels=labs,
                    rng=rng_stream, backend=backend,
                )
                r["grid"] = grid
                r["epsilon"] = eps
                r["dataset"] = dataset_name
                r["trials"] = len(labs)
                r["elapsed_seconds"] = time.time() - t0
                runs.append(r)
                print(
                    f"grid={grid:>3d} eps={eps:<8.1e} {dataset_name:<12} "
                    f"cc={r['correct_consensus_rate']:.4f} "
                    f"fma={r['final_majority_accuracy']:.4f} "
                    f"fails={r['num_failures']:>3d}/{len(labs)} [{r['elapsed_seconds']:.1f}s]"
                )

    args.output.write_text(json.dumps({
        "sid": args.sid,
        "schedule_params": {
            "T_pre": args.T_pre, "T_amp": args.T_amp,
            "T_shake": args.T_shake, "num_shakes": args.num_shakes,
            "T_amp_final": args.T_amp_final,
        },
        "runs": runs,
    }, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
