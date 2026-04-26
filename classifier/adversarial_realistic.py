"""Adversarial structured inputs at REALISTIC density margins.

Instead of density = 0.5 + 1/L^2 (1-cell tipping), use density = 0.5 + delta
for delta in {0.01, 0.02, 0.05}. This matches Fates-era DCP tests.

Input types:
- nearly_stripes_h: horizontal stripes with `delta * L^2` cells flipped
  uniformly at random from 0->1 or 1->0 to hit target density.
- nearly_checker, nearly_block_checker, nearly_half_half: similar.

A UNIVERSAL classifier must handle ALL of these at their target density.
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


def make_stripe_with_density(grid: int, axis: str, density: float, rng: np.random.Generator) -> np.ndarray:
    """Build stripe (axis = 'h' or 'v') then randomly flip cells to reach density."""
    s = np.zeros((grid, grid), dtype=np.uint8)
    if axis == "h":
        s[::2] = 1
    else:
        s[:, ::2] = 1
    current_ones = int(s.sum())
    target_ones = int(round(density * grid * grid))
    diff = target_ones - current_ones
    if diff > 0:
        zero_positions = np.argwhere(s == 0)
        sel = rng.choice(len(zero_positions), size=diff, replace=False)
        for idx in sel:
            r, c = zero_positions[idx]
            s[r, c] = 1
    elif diff < 0:
        one_positions = np.argwhere(s == 1)
        sel = rng.choice(len(one_positions), size=-diff, replace=False)
        for idx in sel:
            r, c = one_positions[idx]
            s[r, c] = 0
    return s


def make_checker_with_density(grid: int, density: float, rng: np.random.Generator) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    s = ((ii + jj) % 2).astype(np.uint8)
    current_ones = int(s.sum())
    target_ones = int(round(density * grid * grid))
    diff = target_ones - current_ones
    if diff > 0:
        z = np.argwhere(s == 0); sel = rng.choice(len(z), size=diff, replace=False)
        for idx in sel:
            r, c = z[idx]; s[r, c] = 1
    elif diff < 0:
        o = np.argwhere(s == 1); sel = rng.choice(len(o), size=-diff, replace=False)
        for idx in sel:
            r, c = o[idx]; s[r, c] = 0
    return s


def make_block_checker_with_density(grid: int, density: float, rng: np.random.Generator) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    s = ((ii // 2 + jj // 2) % 2).astype(np.uint8)
    current_ones = int(s.sum())
    target_ones = int(round(density * grid * grid))
    diff = target_ones - current_ones
    if diff > 0:
        z = np.argwhere(s == 0); sel = rng.choice(len(z), size=diff, replace=False)
        for idx in sel:
            r, c = z[idx]; s[r, c] = 1
    elif diff < 0:
        o = np.argwhere(s == 1); sel = rng.choice(len(o), size=-diff, replace=False)
        for idx in sel:
            r, c = o[idx]; s[r, c] = 0
    return s


def make_half_half_with_density(grid: int, density: float, rng: np.random.Generator) -> np.ndarray:
    s = np.ones((grid, grid), dtype=np.uint8)
    s[:, : grid // 2] = 0
    # Now density is 0.5
    current_ones = int(s.sum())
    target_ones = int(round(density * grid * grid))
    diff = target_ones - current_ones
    if diff > 0:
        z = np.argwhere(s == 0); sel = rng.choice(len(z), size=diff, replace=False)
        for idx in sel:
            r, c = z[idx]; s[r, c] = 1
    elif diff < 0:
        o = np.argwhere(s == 1); sel = rng.choice(len(o), size=-diff, replace=False)
        for idx in sel:
            r, c = o[idx]; s[r, c] = 0
    return s


def build_adversarial_inits(grid: int, densities: list[float], rng: np.random.Generator):
    inits, names, labels = [], [], []
    for d in densities:
        for label_bit in (0, 1):
            actual_d = d if label_bit == 1 else 1.0 - d
            # actual_d > 0.5 means majority=1; < 0.5 means majority=0
            target_label = 1 if actual_d > 0.5 else 0
            for name, fn in [
                ("stripes_h", lambda r=rng: make_stripe_with_density(grid, "h", actual_d, r)),
                ("stripes_v", lambda r=rng: make_stripe_with_density(grid, "v", actual_d, r)),
                ("checker", lambda r=rng: make_checker_with_density(grid, actual_d, r)),
                ("block_checker", lambda r=rng: make_block_checker_with_density(grid, actual_d, r)),
                ("half_half", lambda r=rng: make_half_half_with_density(grid, actual_d, r)),
            ]:
                s = fn()
                inits.append(s.copy())
                names.append(f"{name}_d{actual_d:.3f}")
                labels.append(target_label)
    return np.stack(inits), np.asarray(labels, dtype=np.uint8), names


def run_schedule(
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
    return {
        "consensus_rate": float(consensus.mean()),
        "correct_consensus_rate": float(correct.mean()),
        "num_failures": int((~correct).sum()),
        "per_trial_correct": correct.tolist(),
    }


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
    parser.add_argument("--swaps-mult", type=float, nargs="+", default=[0.0, 4.0, 8.0, 16.0],
                        help="K = mult * L swaps per shake")
    parser.add_argument("--densities", type=float, nargs="+",
                        default=[0.51, 0.52, 0.55],
                        help="Density (majority-color fraction) for adversarial configs")
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 128, 192])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("adversarial_realistic.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    F_bits = catalog.lut_bits[catalog.resolve_rule_ref(args.sid)].astype(np.uint8)
    backend = create_backend(args.backend)

    runs = []
    for grid in args.grids:
        rng_builder = np.random.default_rng(args.seed + grid)
        adv_inits, adv_labels, adv_names = build_adversarial_inits(grid, args.densities, rng_builder)
        for mult in args.swaps_mult:
            K = int(mult * grid)
            rng_stream = np.random.default_rng(args.seed + grid * 100 + int(mult * 1000))
            t0 = time.time()
            r = run_schedule(
                F_bits,
                T_pre=args.T_pre, T_amp=args.T_amp,
                T_shake=args.T_shake, num_shakes=args.num_shakes,
                T_amp_final=args.T_amp_final,
                swaps_per_shake=K,
                initial_states=adv_inits, labels=adv_labels,
                rng=rng_stream, backend=backend,
            )
            r["grid"] = grid
            r["swaps_mult"] = mult
            r["swaps_per_shake"] = K
            r["elapsed_seconds"] = time.time() - t0
            runs.append(r)
            correct_arr = np.asarray(r["per_trial_correct"])
            print(
                f"grid={grid:>3d} mult={mult:<4.1f} K={K:>5d} "
                f"cc={r['correct_consensus_rate']:.3f} "
                f"fails={r['num_failures']}/{len(adv_labels)} "
                f"[{r['elapsed_seconds']:.1f}s]"
            )
            # Group failures by pattern type and density
            if r['num_failures'] > 0:
                for i, name in enumerate(adv_names):
                    if not correct_arr[i]:
                        print(f"    FAIL: {name}  label={adv_labels[i]}")

    args.output.write_text(json.dumps({
        "sid": args.sid,
        "densities": args.densities,
        "runs": runs,
    }, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
