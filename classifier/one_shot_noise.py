"""Q2 follow-up: Does a SINGLE random perturbation suffice?

Schedule:
    F^T_pre -> [apply ONE step: F(state) XOR Bernoulli(epsilon)] ->
    moore81^T_amp_final

One and only one random step. If this closes the adversarial gap at the same
random-input 100% level, we've reduced the stochastic part to its absolute
minimum. In the asymptotic L->inf regime, expected number of random flips
is epsilon * L^2, so with epsilon = C/L^2 we get O(1) expected flips. The
question: what's the smallest C that works?
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


def run_one_shot(
    F_bits: np.ndarray,
    *,
    T_pre: int,
    T_amp_pre: int,
    num_noise_cells: int,  # expected # cells flipped per trial in the single-shot noise
    T_amp_final: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    backend,
) -> dict:
    import mlx.core as mx
    batch, H, W = initial_states.shape
    L2 = H * W
    states = backend.asarray(initial_states, dtype="uint8")
    tiled_F = backend.asarray(np.tile(F_bits, (batch, 1)), dtype="uint8")

    # Pre-phase: deterministic F
    for _ in range(T_pre):
        states = apply_lut_mlx(states, tiled_F)
    # Pre-amp
    for _ in range(T_amp_pre):
        states = apply_radius_step_mlx(states, "moore81")
    # Single-shot noise: flip num_noise_cells sites per trial, uniformly at random
    if num_noise_cells > 0:
        s_np = backend.to_numpy(states)
        for bidx in range(batch):
            flat = rng.choice(L2, size=num_noise_cells, replace=False)
            for idx in flat:
                y = idx // W; x = idx % W
                s_np[bidx, y, x] ^= 1
        states = backend.asarray(s_np, dtype="uint8")
    # Final amp
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
    parser.add_argument("--T-amp-pre", type=int, default=64)
    parser.add_argument("--T-amp-final", type=int, default=1024)
    parser.add_argument("--noise-cells-per-L2-pct", type=float, nargs="+",
                        default=[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0])
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--trials-per-side", type=int, default=128)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("one_shot_noise.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    F_bits = catalog.lut_bits[catalog.resolve_rule_ref(args.sid)].astype(np.uint8)
    backend = create_backend(args.backend)

    runs = []
    for grid in args.grids:
        L2 = grid * grid
        rng_init = np.random.default_rng(args.seed + grid)
        random_inits, random_labels = build_random_inits(rng_init, grid, args.probabilities, args.trials_per_side)
        adv_inits, adv_labels, adv_names = build_adversarial_inits(grid)

        for pct in args.noise_cells_per_L2_pct:
            num_noise_cells = int(round(pct / 100.0 * L2))
            for dataset_name, inits, labs in [("random", random_inits, random_labels),
                                                ("adversarial", adv_inits, adv_labels)]:
                rng_stream = np.random.default_rng(args.seed + grid * 100 + int(pct * 1e6) + len(labs))
                t0 = time.time()
                r = run_one_shot(
                    F_bits,
                    T_pre=args.T_pre, T_amp_pre=args.T_amp_pre,
                    num_noise_cells=num_noise_cells,
                    T_amp_final=args.T_amp_final,
                    initial_states=inits, labels=labs,
                    rng=rng_stream, backend=backend,
                )
                r["grid"] = grid
                r["noise_cells"] = num_noise_cells
                r["noise_pct"] = pct
                r["dataset"] = dataset_name
                r["trials"] = len(labs)
                r["elapsed_seconds"] = time.time() - t0
                runs.append(r)
                print(
                    f"grid={grid:>3d} cells={num_noise_cells:>6d} "
                    f"pct={pct:<6.3f} {dataset_name:<12} "
                    f"cc={r['correct_consensus_rate']:.4f} "
                    f"fma={r['final_majority_accuracy']:.4f} "
                    f"fails={r['num_failures']:>3d}/{len(labs)} [{r['elapsed_seconds']:.1f}s]"
                )

    args.output.write_text(json.dumps({
        "sid": args.sid, "runs": runs,
    }, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
