"""GPU-batched stripe-disruption probe for candidate R_break rules.

For each candidate rule k, we test how aggressively it disrupts a
horizontal-stripe lattice when applied on its own.  Pure-stripe lattices
have alternating-row structure, so the vertical-neighbour disagreement
rate is exactly 1.0.  After applying a candidate rule for T_probe steps:

- stripe-preserving rules leave the disagreement rate near 1.0;
- stripe-breaking rules drive it toward 0.5 (random-like).

We additionally probe horizontal-stripe AND vertical-stripe AND
2x2-block-checkerboard inputs and combine the three scores: a strong
R_break should disrupt all three structural fixed-point families.

The combined score is the maximum (over the three input types) of the
post-probe disagreement-rate distance from 0.5; LOW score = strongly
disruptive on every input type; HIGH score = preserves at least one
structure.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# --- repo-root path bootstrap ---
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
_CATALOG_DIR = _REPO_ROOT / "catalogs"
# --------------------------------

from ca_search.simulator import create_backend
from witek_sampler import WitekIndex, sample_random_lut_batch


def _build_stripe_h(L: int) -> np.ndarray:
    s = np.zeros((L, L), dtype=np.uint8)
    s[::2] = 1
    return s


def _build_stripe_v(L: int) -> np.ndarray:
    s = np.zeros((L, L), dtype=np.uint8)
    s[:, ::2] = 1
    return s


def _build_block_checker(L: int) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    return ((ii // 2 + jj // 2) % 2).astype(np.uint8)


def _vertical_disagree(states_mlx):
    """For (B, H, W) uint8, return (B,) float = mean(|s[i,j] - s[i+1,j]|)."""
    import mlx.core as mx
    rolled = mx.roll(states_mlx, -1, axis=1)
    diff = (states_mlx != rolled).astype(mx.uint8)
    return mx.mean(diff, axis=(1, 2))


def _horizontal_disagree(states_mlx):
    import mlx.core as mx
    rolled = mx.roll(states_mlx, -1, axis=2)
    diff = (states_mlx != rolled).astype(mx.uint8)
    return mx.mean(diff, axis=(1, 2))


def _block_disagree(states_mlx):
    """Compute fraction of 2x2 blocks that *aren't* a 2x2 single-color block.
    A perfect 2x2-block-checker has every 2x2 aligned block solid, so this is 0.
    A random lattice has it close to 1 - 2*(0.5^4) = 1 - 0.125 = 0.875."""
    import mlx.core as mx
    # Consider all 2x2 windows by rolling.
    s00 = states_mlx
    s01 = mx.roll(states_mlx, -1, axis=2)
    s10 = mx.roll(states_mlx, -1, axis=1)
    s11 = mx.roll(mx.roll(states_mlx, -1, axis=1), -1, axis=2)
    same_all = ((s00 == s01) & (s00 == s10) & (s00 == s11)).astype(mx.uint8)
    # Perfect 2x2-block-checker on aligned 2x2 grid: at the 4 aligned positions
    # all-same; at offset positions, the 2x2 window straddles two color blocks.
    # We just compute the "rough" disagreement rate over all positions.
    return 1.0 - mx.mean(same_all, axis=(1, 2))


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


def step_pairwise_batch(states_mlx, rules_mlx):
    """Apply a different LUT to each lattice in the batch.

    states_mlx: (B, H, W) uint8 mlx
    rules_mlx:  (B, 512) uint8 mlx
    """
    import mlx.core as mx
    indices = _pair_indices_mlx(states_mlx)
    out = mx.take_along_axis(rules_mlx[:, None, None, :], indices[..., None], axis=3)
    return mx.squeeze(out, axis=3).astype(mx.uint8)


def probe_rules(
    luts: np.ndarray,           # (K, 512) uint8
    *,
    L: int = 64,
    T_probe: int = 64,
    p_majority: float = 0.05,
    backend=None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Stochastic-mixture probe: each candidate rule R_k is run in a
    per-cell-per-step mixture with Moore-9 majority M.  At each step,
    each cell chooses R_k with probability (1 - p_majority) or M with
    probability p_majority.  This breaks shift-invariance the same way
    the strict-CA classifier does, so structural fixed points of R_k
    that survive shift-invariance can still be visibly degraded by the
    occasional M perturbation.

    Score per input = max(|h_disagree - 0.5|, |v_disagree - 0.5|)
    measured at the END of T_probe steps.  Combined score = max across
    inputs.  LOW combined score = candidate disrupts every structural
    fixed-point family toward random-like behaviour.

    Note: the M perturbations at each step are not number-conserving,
    so post-probe density may drift away from 0.5; for stripe and
    checker inputs (initial density 0.5), strong density drift to 0/1
    corresponds to consensus, also reported.
    """
    import mlx.core as mx
    if backend is None:
        backend = create_backend("mlx")
    if rng is None:
        rng = np.random.default_rng(0)
    K = luts.shape[0]

    rules_mlx = backend.asarray(luts, dtype="uint8")
    M = np.asarray(_moore_majority_lut(), dtype=np.uint8)
    M_mlx = backend.asarray(np.broadcast_to(M, (K, 512)).copy(), dtype="uint8")

    results = {}
    per_input_scores = []
    for input_name, build_fn in [
        ("stripes_h", _build_stripe_h),
        ("stripes_v", _build_stripe_v),
        ("block_checker", _build_block_checker),
    ]:
        init = build_fn(L)
        init_batch = np.broadcast_to(init, (K, L, L)).copy()
        states = backend.asarray(init_batch, dtype="uint8")

        for _ in range(T_probe):
            # Apply both rules to current state, then choose per cell.
            r_out = step_pairwise_batch(states, rules_mlx)
            m_out = step_pairwise_batch(states, M_mlx)
            mask_np = (rng.random((K, L, L)) < p_majority).astype(np.uint8)
            mask = backend.asarray(mask_np, dtype="uint8")
            states = (mask * m_out + (1 - mask) * r_out).astype(mx.uint8)

        h_d = _horizontal_disagree(states)
        v_d = _vertical_disagree(states)
        density = mx.mean(states.astype(mx.float32), axis=(1, 2))
        score = mx.maximum(mx.abs(h_d - 0.5), mx.abs(v_d - 0.5))
        results[input_name + "_h_disagree"] = backend.to_numpy(h_d)
        results[input_name + "_v_disagree"] = backend.to_numpy(v_d)
        results[input_name + "_density"] = backend.to_numpy(density)
        results[input_name + "_score"] = backend.to_numpy(score)
        per_input_scores.append(backend.to_numpy(score))

    combined = np.maximum.reduce(per_input_scores)
    results["combined"] = combined
    return results


def _moore_majority_lut():
    bits = [0] * 512
    for i in range(512):
        bits[i] = 1 if bin(i).count("1") >= 5 else 0
    return bits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10_000,
                        help="Number of rules to sample from the witek corpus")
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--T-probe", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--p-majority", type=float, default=0.05,
                        help="Probability of applying Moore-9 majority per cell per step")
    parser.add_argument("--top-n", type=int, default=200,
                        help="Persist top-N (lowest combined score) rules")
    parser.add_argument("--output", type=Path,
                        default=Path("results/density_classification/2026-04-29/r_break_probe.json"))
    args = parser.parse_args()

    print(f"Sampling {args.K} random rules from witek/ ...")
    t0 = time.time()
    luts, global_idx = sample_random_lut_batch(
        args.K, seed=args.seed, return_global_indices=True,
    )
    t_sample = time.time() - t0
    print(f"  sampled in {t_sample*1000:.1f}ms; LUTs shape={luts.shape}")

    print(f"Running stochastic-mixture stripe-disruption probe (L={args.L}, T={args.T_probe}, p_M={args.p_majority}) ...")
    t0 = time.time()
    rng_probe = np.random.default_rng(args.seed * 17)
    out = probe_rules(luts, L=args.L, T_probe=args.T_probe,
                       p_majority=args.p_majority, rng=rng_probe)
    t_probe = time.time() - t0
    print(f"  probe done in {t_probe:.2f}s ({args.K/t_probe:.0f} rules/s)")

    combined = out["combined"]
    order = np.argsort(combined)  # ascending: lowest first

    # Top-N to persist
    top_n = order[:args.top_n]
    persistable = []
    for j in top_n:
        persistable.append({
            "global_index": int(global_idx[j]),
            "lut_hex": "".join(f"{b:01x}" for b in luts[j]),
            "stripes_h_score": float(out["stripes_h_score"][j]),
            "stripes_v_score": float(out["stripes_v_score"][j]),
            "block_checker_score": float(out["block_checker_score"][j]),
            "combined_score": float(out["combined"][j]),
            "stripes_h_h_disagree": float(out["stripes_h_h_disagree"][j]),
            "stripes_h_v_disagree": float(out["stripes_h_v_disagree"][j]),
            "stripes_v_h_disagree": float(out["stripes_v_h_disagree"][j]),
            "stripes_v_v_disagree": float(out["stripes_v_v_disagree"][j]),
            "block_checker_h_disagree": float(out["block_checker_h_disagree"][j]),
            "block_checker_v_disagree": float(out["block_checker_v_disagree"][j]),
        })

    print()
    print(f"Top {min(20, args.top_n)} R_break candidates (lowest combined score):")
    print(f"  {'rank':<5} {'global_idx':>11} {'comb':>6} {'h_sc':>6} {'v_sc':>6} {'bc_sc':>6}")
    for rank, j in enumerate(top_n[:20]):
        print(f"  {rank:<5} {int(global_idx[j]):>11} "
              f"{out['combined'][j]:>6.3f} "
              f"{out['stripes_h_score'][j]:>6.3f} "
              f"{out['stripes_v_score'][j]:>6.3f} "
              f"{out['block_checker_score'][j]:>6.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "K": args.K, "L": args.L, "T_probe": args.T_probe, "seed": args.seed,
        "n_rules_kept": len(persistable),
        "top_n_rules": persistable,
        "score_histogram": {
            "min": float(combined.min()),
            "p25": float(np.percentile(combined, 25)),
            "p50": float(np.percentile(combined, 50)),
            "p75": float(np.percentile(combined, 75)),
            "max": float(combined.max()),
        },
        "elapsed_seconds": {"sample": t_sample, "probe": t_probe},
    }, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
