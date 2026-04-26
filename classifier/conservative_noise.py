"""Density-conserving stochastic perturbation.

Instead of independent per-cell flips (which alter global density), use
random SWAPS: pick K random pairs of cells and swap their values. This
preserves the exact particle count per realization.

Rationale: our tipping-cell bias is only ONE cell out of L^2. Per-cell
flip noise at rate epsilon introduces ~epsilon*L^2 flips, overwhelming the
bias. Swaps don't change global count at all, so the bias is preserved.

Schedule:
    F^T_pre  [ moore81^T_amp  F_shake^T_shake  swap^K ]^{num_shakes}
    moore81^T_amp_final

K is the number of swaps per shake-cycle. K can scale with L (e.g. K = L
means ~1 swap per row, enough to break row-independence).
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


def apply_swaps(states_np: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Apply K random swaps per trial. Density-conserving exactly."""
    if K <= 0:
        return states_np
    batch, H, W = states_np.shape
    L2 = H * W
    out = states_np.copy()
    for bidx in range(batch):
        # Pick K pairs (2K indices). Use numpy for CPU-side scatter.
        idx_a = rng.integers(0, L2, size=K)
        idx_b = rng.integers(0, L2, size=K)
        # Skip self-swaps; also skip same-value swaps (no effect)
        flat = out[bidx].ravel()
        va = flat[idx_a].copy()
        vb = flat[idx_b].copy()
        flat[idx_a] = vb
        flat[idx_b] = va
        out[bidx] = flat.reshape(H, W)
    return out


def run_schedule_swaps(
    F_bits: np.ndarray,
    *,
    T_pre: int,
    T_amp: int,
    T_shake: int,
    num_shakes: int,
    T_amp_final: int,
    swaps_per_shake: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    backend,
) -> dict:
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
        # Apply swaps (density-conserving noise)
        if swaps_per_shake > 0:
            s_np = backend.to_numpy(states)
            s_np = apply_swaps(s_np, swaps_per_shake, rng)
            states = backend.asarray(s_np, dtype="uint8")

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


def build_adversarial_inits(grid: int):
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
    parser.add_argument("--swaps-scaling", choices=["fixed", "L", "L2", "LlogL"], default="L")
    parser.add_argument("--swaps-values", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
                        help="Multiplier on L (or L^2) for number of swaps per shake")
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 128, 192])
    parser.add_argument("--trials-per-side", type=int, default=128)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("conservative_noise.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    F_bits = catalog.lut_bits[catalog.resolve_rule_ref(args.sid)].astype(np.uint8)
    backend = create_backend(args.backend)

    runs = []
    for grid in args.grids:
        rng_init = np.random.default_rng(args.seed + grid)
        random_inits, random_labels = build_random_inits(rng_init, grid, args.probabilities, args.trials_per_side)
        adv_inits, adv_labels, adv_names = build_adversarial_inits(grid)

        for mult in args.swaps_values:
            if args.swaps_scaling == "fixed":
                K = int(mult)
            elif args.swaps_scaling == "L":
                K = int(mult * grid)
            elif args.swaps_scaling == "L2":
                K = int(mult * grid * grid)
            elif args.swaps_scaling == "LlogL":
                K = int(mult * grid * np.log2(max(grid, 2)))
            for dataset_name, inits, labs in [("random", random_inits, random_labels),
                                                ("adversarial", adv_inits, adv_labels)]:
                rng_stream = np.random.default_rng(args.seed + grid * 100 + int(mult * 1000) + len(labs))
                t0 = time.time()
                r = run_schedule_swaps(
                    F_bits,
                    T_pre=args.T_pre, T_amp=args.T_amp,
                    T_shake=args.T_shake, num_shakes=args.num_shakes,
                    T_amp_final=args.T_amp_final,
                    swaps_per_shake=K,
                    initial_states=inits, labels=labs,
                    rng=rng_stream, backend=backend,
                )
                r["grid"] = grid
                r["swaps_mult"] = mult
                r["swaps_per_shake"] = K
                r["dataset"] = dataset_name
                r["trials"] = len(labs)
                r["elapsed_seconds"] = time.time() - t0
                runs.append(r)
                print(
                    f"grid={grid:>3d} mult={mult:<5.1f} K={K:>6d} {dataset_name:<12} "
                    f"cc={r['correct_consensus_rate']:.4f} "
                    f"fma={r['final_majority_accuracy']:.4f} "
                    f"fails={r['num_failures']:>3d}/{len(labs)} [{r['elapsed_seconds']:.1f}s]"
                )

    args.output.write_text(json.dumps({
        "sid": args.sid,
        "swaps_scaling": args.swaps_scaling,
        "runs": runs,
    }, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
