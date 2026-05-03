"""Strict-CA stochastic density classifier.

The schedule has *no* non-local operations: every step is the
synchronous application of a binary cellular automaton rule chosen
independently per cell from a small mixture catalogue.  At step t and
site (i, j), with probabilities (p_1, ..., p_K) summing to 1, the
local rule index r(t, i, j) is drawn from {1, ..., K}, and cell
(i, j) is updated as

    s'(i, j) = R_{r(t, i, j)}(local neighborhood at (i, j)).

Each R_k is a Moore-9 lookup table (a standard binary CA rule).
The randomness is local (one Bernoulli/categorical draw per cell per
step) and the rule application is local.  No global state crosses the
lattice; the only way information can spread is through repeated rule
application, exactly as for a classical CA.

We expose two evaluation modes:
- 'fixed_steps': run for a given number of steps and report final
  consensus / density.
- 'until_consensus': run until the lattice reaches all-0 or all-1, with
  a maximum step cap; report time-to-consensus.

The reference 'baseline' (non-local) classifier is the L-scaled swap
schedule from classifier/scaled_schedule.py.  Comparing time-to-
consensus on identical inputs quantifies how much we lose by being
strictly local.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# --- repo-root path bootstrap (BEFORE ca_search/local imports) ---
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
_CATALOG_DIR = _REPO_ROOT / "catalogs"
# ----------------------------------------------------------------

from amplifier_library import build_amplifier
from ca_search.binary_catalog import load_binary_catalog
from ca_search.density_classification import (
    embedded_diagonal_traffic_rule_bits,
    embedded_diagonal_traffic_rule_bits as _imported_diag,  # alias for clarity
    moore_majority_rule_bits,
    von_neumann_majority_rule_bits,
)
from ca_search.lut import (
    embedded_von_neumann_traffic_rule_bits,
    rigid_rule_bits,
    identity_rule_bits,
)
from ca_search.simulator import create_backend


# ---------------------------------------------------------------------------
# Rule registry: any callable returning a 512-entry uint8 LUT.
# ---------------------------------------------------------------------------

def _identity():
    return identity_rule_bits()


def _moore_majority():
    return moore_majority_rule_bits()


def _vn_majority():
    return von_neumann_majority_rule_bits()


def _traffic_vn(velocity: str):
    return embedded_von_neumann_traffic_rule_bits(velocity)


def _traffic_diag(velocity: str):
    return embedded_diagonal_traffic_rule_bits(velocity)


def _shift(velocity: str):
    return rigid_rule_bits({
        "north": "y", "south": "b", "east": "w", "west": "t",
        "northeast": "z", "northwest": "x",
        "southeast": "c", "southwest": "a",
    }[velocity])


BUILTIN_RULES = {
    "identity": _identity,
    "moore_maj": _moore_majority,
    "vn_maj": _vn_majority,
    "traffic_n": lambda: _traffic_vn("north"),
    "traffic_s": lambda: _traffic_vn("south"),
    "traffic_e": lambda: _traffic_vn("east"),
    "traffic_w": lambda: _traffic_vn("west"),
    "traffic_ne": lambda: _traffic_diag("northeast"),
    "traffic_nw": lambda: _traffic_diag("northwest"),
    "traffic_se": lambda: _traffic_diag("southeast"),
    "traffic_sw": lambda: _traffic_diag("southwest"),
    "shift_n": lambda: _shift("north"),
    "shift_s": lambda: _shift("south"),
    "shift_e": lambda: _shift("east"),
    "shift_w": lambda: _shift("west"),
}


def resolve_rule_bits(name: str, *, catalog=None) -> list[int]:
    """Return a 512-entry LUT for `name`.  If name starts with 'sid:' or
    'legacy:' or 'rank:' or 'stable:', resolve via the binary catalog."""
    if name in BUILTIN_RULES:
        return BUILTIN_RULES[name]()
    if catalog is None:
        raise ValueError(f"unknown built-in rule '{name}' and no catalog provided")
    idx = catalog.resolve_rule_ref(name)
    return list(catalog.lut_bits[idx].astype(np.uint8))


# ---------------------------------------------------------------------------
# Mixture step.
# ---------------------------------------------------------------------------

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


def step_mixture_mlx(states, rule_luts_mlx, probs_cdf_np, rng):
    """Apply each rule's LUT to ALL cells, then gather per-cell using a
    random index drawn from the categorical probs.  Local-only:
    randomness is per cell, per step, independent.

    states         : (batch, H, W) uint8 mlx array
    rule_luts_mlx  : (K, 512) uint8 mlx array (LUT bits per rule)
    probs_cdf_np   : (K,) float64 numpy array, cumulative probabilities
                     (last entry must be 1.0)
    rng            : np.random.Generator
    """
    import mlx.core as mx
    batch, H, W = states.shape
    K = rule_luts_mlx.shape[0]

    indices = _pair_indices_mlx(states)  # (batch, H, W) uint16

    # Apply EVERY rule's LUT to every cell, get (batch, H, W, K) of outputs.
    # rule_luts_mlx[k, :] is the LUT for rule k. We gather indices[:, :, :]
    # into each LUT row.  Easiest: replicate across the K axis.
    # Shape engineering: rule_luts_mlx[K, 512] -> per-rule output via
    # take_along_axis on indices broadcast to (K, batch, H, W).
    # We do K separate LUT lookups (small K) and stack:
    outputs = []
    for k in range(K):
        lut_k = rule_luts_mlx[k]  # (512,)
        # take_along_axis equivalent: simple gather since indices is uint16.
        out_k = lut_k[indices.astype(mx.int32)]  # (batch, H, W)
        outputs.append(out_k)
    out_stack = mx.stack(outputs, axis=0)  # (K, batch, H, W)

    # Per-cell random rule index.
    # Sample uniform in [0,1) and threshold by CDF.
    u = rng.random((batch, H, W))
    chosen = np.zeros((batch, H, W), dtype=np.int32)
    for k in range(K - 1, -1, -1):
        chosen = np.where(u < probs_cdf_np[k], k, chosen)
    # Note: np.where(condition, a, b) returns b where condition false. We
    # iterate top-down so the lowest-k satisfying u<cdf[k] wins. (cdf[0] is
    # smallest; cdf[K-1]=1.0).

    chosen_mlx = mx.array(chosen)  # (batch, H, W)
    # Gather: out[b, i, j] = out_stack[chosen[b,i,j], b, i, j].
    # MLX doesn't have a simple gather-along-axis-0 for arbitrary index, so
    # we use take + arithmetic.  Implement via one-hot multiply:
    #   one_hot[K, batch, H, W] = (chosen == k)
    #   out = sum_k one_hot[k] * out_stack[k]
    # K is small (<=4 in our experiments), so the loop is cheap.
    final = mx.zeros_like(out_stack[0])
    for k in range(K):
        mask_k_np = (chosen == k).astype(np.uint8)
        mask_k = mx.array(mask_k_np)
        final = final + mask_k * out_stack[k]
    return final.astype(mx.uint8)


def run_strict_ca(
    initial_states: np.ndarray,
    *,
    rule_lut_list: list[list[int]],
    probabilities: list[float],
    max_steps: int,
    until_consensus: bool,
    backend,
    rng: np.random.Generator,
    record_density_every: int = 0,
) -> dict:
    """Run the strict-CA mixture classifier on a batch of inputs.

    Returns:
        dict with keys final_states, time_to_consensus_per_trial,
        density_history (only if record_density_every > 0).
    """
    import mlx.core as mx

    batch, H, W = initial_states.shape
    L2 = H * W

    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1, got {sum(probabilities)}")
    probs_cdf = np.cumsum(probabilities, dtype=np.float64)
    probs_cdf[-1] = 1.0  # numerical safety

    # Build per-rule tiled LUTs once.
    K = len(rule_lut_list)
    rule_luts_arr = np.asarray(rule_lut_list, dtype=np.uint8)  # (K, 512)
    rule_luts_mlx = backend.asarray(rule_luts_arr, dtype="uint8")

    states = backend.asarray(initial_states, dtype="uint8")

    # Track per-trial first-step at which consensus was reached.
    time_to_consensus = [None] * batch
    final_correct = [None] * batch
    density_log: list[list[float]] = []

    for step in range(max_steps):
        states = step_mixture_mlx(states, rule_luts_mlx, probs_cdf, rng)
        cur = backend.to_numpy(states)
        totals = cur.reshape(batch, -1).sum(axis=1)

        if record_density_every > 0 and step % record_density_every == 0:
            density_log.append([float(t) / L2 for t in totals])

        # Mark trials that reached consensus this step.
        for i in range(batch):
            if time_to_consensus[i] is not None:
                continue
            if totals[i] == 0 or totals[i] == L2:
                time_to_consensus[i] = step + 1

        if until_consensus and all(t is not None for t in time_to_consensus):
            break

    # Fill in remainder.
    final = backend.to_numpy(states)
    final_totals = final.reshape(batch, -1).sum(axis=1)
    return {
        "final_states": final,
        "final_totals": final_totals.tolist(),
        "time_to_consensus": time_to_consensus,
        "density_log": density_log,
        "max_steps": max_steps,
        "n_rules": len(rule_lut_list),
        "probabilities": list(probabilities),
    }


# ---------------------------------------------------------------------------
# Test-suite construction & scoring (compatible with adversarial_realistic).
# ---------------------------------------------------------------------------

def _correct_consensus(final_totals: list[int], labels: np.ndarray, L2: int) -> np.ndarray:
    final = np.asarray(final_totals)
    return ((final == 0) & (labels == 0)) | ((final == L2) & (labels == 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path,
                        default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--rules", nargs="+", required=True,
                        help="Rule names (built-in or sid:...) for the mixture")
    parser.add_argument("--probabilities", type=float, nargs="+", required=True,
                        help="Mixture probabilities, must sum to 1 and match --rules length")
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--probability", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--n-trials-per-side", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=8192)
    parser.add_argument("--until-consensus", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("strict_ca_run.json"))
    args = parser.parse_args()

    if len(args.rules) != len(args.probabilities):
        raise SystemExit("--rules and --probabilities must have same length")
    if abs(sum(args.probabilities) - 1.0) > 1e-6:
        raise SystemExit(f"probabilities must sum to 1, got {sum(args.probabilities)}")

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    rule_luts = []
    for name in args.rules:
        bits = resolve_rule_bits(name, catalog=catalog)
        rule_luts.append(bits)

    backend = create_backend(args.backend)

    # Build batch: random near-critical inputs at exact density.
    rng_init = np.random.default_rng(args.seed + args.grid)
    inits = []
    labs = []
    L = args.grid
    for p in args.probability:
        n_total = L * L
        n_ones = int(round(p * n_total))
        for _ in range(args.n_trials_per_side):
            flat = np.zeros(n_total, dtype=np.uint8); flat[:n_ones] = 1
            rng_init.shuffle(flat)
            inits.append(flat.reshape(L, L))
            labs.append(1 if p > 0.5 else 0)
    init_states = np.stack(inits)
    labels = np.asarray(labs, dtype=np.uint8)

    rng_run = np.random.default_rng(args.seed * 17 + args.grid)
    t0 = time.time()
    out = run_strict_ca(
        init_states,
        rule_lut_list=rule_luts,
        probabilities=args.probabilities,
        max_steps=args.max_steps,
        until_consensus=args.until_consensus,
        backend=backend, rng=rng_run,
        record_density_every=0,
    )
    elapsed = time.time() - t0

    correct = _correct_consensus(out["final_totals"], labels, L * L)
    n = len(labels)
    n_consensus = sum(1 for t in out["time_to_consensus"] if t is not None)
    n_correct = int(correct.sum())
    times = [t for t in out["time_to_consensus"] if t is not None]
    median_time = float(np.median(times)) if times else float("nan")
    p95_time = float(np.percentile(times, 95)) if times else float("nan")
    max_time = max(times) if times else 0

    summary = {
        "rules": args.rules,
        "probabilities": args.probabilities,
        "grid": L,
        "n_trials": n,
        "n_consensus": n_consensus,
        "consensus_rate": n_consensus / n,
        "correct_consensus_rate": n_correct / n,
        "median_steps_to_consensus": median_time,
        "p95_steps_to_consensus": p95_time,
        "max_steps_to_consensus": max_time,
        "elapsed_seconds": elapsed,
        "max_steps": args.max_steps,
    }
    print(json.dumps(summary, indent=2))
    args.output.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
