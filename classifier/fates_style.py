"""2D Fates-style probabilistic / stochastic density classifier.

At each time step, at each lattice site independently, pick one of:
- the preprocessor NCCA F (with probability p_F)
- the local majority amplifier M (with probability 1 - p_F)

and apply that rule's output. Synchronous update across the lattice, but the
CHOICE of rule is random per site per step. This breaks translational
invariance (different sites get different rules at the same step) while
keeping each rule deterministic and local.

Density is NOT strictly conserved step-by-step (because M is not NCCA), but
the classifier still converges toward consensus on the correct global
majority. The tuning knob is p_F:
- low p_F  -> majority dominates, fast convergence, may lock at wrong state
- high p_F -> NCCA dominates, slow convergence, better classification in the
              stochastic sense

Modes:
- 'mix': per-site Bernoulli(p_F) choice of F vs M every step.
- 'async_pre_then_det': asynchronous F phase (each step, a random fraction
  alpha of cells apply F, others keep value) for T1 steps, then deterministic
  majority for T2 steps.
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
    build_amplifier,
)
from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend


RADIUS_KINDS = set(RADIUS_KIND_SPECS.keys())


def step_mix_mlx(states, F_tiled, M_name, p_F, rng_key_source):
    """One step of 'mix' mode on an MLX array.

    states: (batch, H, W) uint8 mlx array.
    F_tiled: (batch, 512) uint8 mlx array (tiled F rule bits).
    M_name: amplifier kind (moore9/moore81/etc.).
    p_F: probability of using F at any site.
    rng_key_source: numpy Generator producing the mask per call.
    """
    import mlx.core as mx

    batch, H, W = states.shape
    # apply F to all cells (via backend step) and M to all cells; combine
    # with a random mask.
    # For F, we need the full Moore step via the standard simulator logic —
    # use numpy path here since we only need F in parallel with M.
    # MLX step_pairwise for F
    x = mx.roll(mx.roll(states, 1, axis=1), 1, axis=2)
    y = mx.roll(states, 1, axis=1)
    z = mx.roll(mx.roll(states, 1, axis=1), -1, axis=2)
    t = mx.roll(states, 1, axis=2)
    u = states
    w = mx.roll(states, -1, axis=2)
    a = mx.roll(mx.roll(states, -1, axis=1), 1, axis=2)
    b = mx.roll(states, -1, axis=1)
    c = mx.roll(mx.roll(states, -1, axis=1), -1, axis=2)
    indices = (
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
    F_out = mx.take_along_axis(F_tiled[:, None, None, :], indices[..., None], axis=3)
    F_out = mx.squeeze(F_out, axis=3).astype(mx.uint8)

    # apply M via the radius-majority helper
    M_out = apply_radius_step_mlx(states, M_name) if M_name in RADIUS_KINDS else None
    if M_name in RADIUS_KINDS:
        pass
    else:
        # 3x3 sub-majority: build its LUT and apply as pairwise step
        bits = np.asarray(build_amplifier(M_name), dtype=np.uint8)
        tiled = mx.array(np.tile(bits, (batch, 1)))
        M_out = mx.take_along_axis(tiled[:, None, None, :], indices[..., None], axis=3)
        M_out = mx.squeeze(M_out, axis=3).astype(mx.uint8)

    # Random mask: 1 where we pick F, 0 where we pick M
    mask_np = (rng_key_source.random((batch, H, W)) < p_F).astype(np.uint8)
    mask = mx.array(mask_np)

    out = mx.where(mask == 1, F_out, M_out)
    return out.astype(mx.uint8)


def run_fates_mix(
    F_bits: np.ndarray,
    M_name: str,
    *,
    p_F: float,
    total_steps: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    backend,
    rng: np.random.Generator,
) -> dict:
    batch, H, W = initial_states.shape
    L2 = H * W
    states = backend.asarray(initial_states, dtype="uint8")
    F_tiled = backend.asarray(np.tile(F_bits, (batch, 1)), dtype="uint8")

    for step in range(total_steps):
        states = step_mix_mlx(states, F_tiled, M_name, p_F, rng)

    final = backend.to_numpy(states)
    totals = final.reshape(batch, -1).sum(axis=1)
    all_zero = (totals == 0)
    all_one = (totals == L2)
    consensus = all_zero | all_one
    correct = (all_zero & (labels == 0)) | (all_one & (labels == 1))
    final_majority = (totals > (L2 / 2)).astype(np.uint8)
    return {
        "total_steps": total_steps,
        "consensus_rate": float(consensus.mean()),
        "correct_consensus_rate": float(correct.mean()),
        "final_majority_accuracy": float((final_majority == labels).mean()),
        "final_density_mean": float(final.mean()),
    }


def run_fates_mix_with_cutoff(
    F_bits: np.ndarray,
    M_name: str,
    *,
    p_F: float,
    max_steps: int,
    final_majority_steps: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    backend,
    rng: np.random.Generator,
) -> dict:
    """Run stochastic mix for up to max_steps, then deterministic final majority
    for final_majority_steps steps to 'collapse' to consensus.
    """
    batch, H, W = initial_states.shape
    L2 = H * W
    states = backend.asarray(initial_states, dtype="uint8")
    F_tiled = backend.asarray(np.tile(F_bits, (batch, 1)), dtype="uint8")

    for step in range(max_steps):
        states = step_mix_mlx(states, F_tiled, M_name, p_F, rng)
    for step in range(final_majority_steps):
        if M_name in RADIUS_KINDS:
            states = apply_radius_step_mlx(states, M_name)
        else:
            bits = np.asarray(build_amplifier(M_name), dtype=np.uint8)
            tiled = backend.asarray(np.tile(bits, (batch, 1)), dtype="uint8")
            states = backend.step_pairwise(states, tiled)

    final = backend.to_numpy(states)
    totals = final.reshape(batch, -1).sum(axis=1)
    all_zero = (totals == 0)
    all_one = (totals == L2)
    consensus = all_zero | all_one
    correct = (all_zero & (labels == 0)) | (all_one & (labels == 1))
    final_majority = (totals > (L2 / 2)).astype(np.uint8)
    return {
        "total_steps": max_steps + final_majority_steps,
        "consensus_rate": float(consensus.mean()),
        "correct_consensus_rate": float(correct.mean()),
        "final_majority_accuracy": float((final_majority == labels).mean()),
        "final_density_mean": float(final.mean()),
    }


def build_random_inits(rng, grid, probabilities, trials_per_side):
    inits, labs, prob_col = [], [], []
    for p in probabilities:
        init = (rng.random((trials_per_side, grid, grid)) < p).astype(np.uint8)
        totals = init.reshape(trials_per_side, -1).sum(axis=1)
        lab = (totals > (grid * grid / 2)).astype(np.uint8)
        inits.append(init)
        labs.append(lab)
        prob_col.extend([p] * trials_per_side)
    return np.concatenate(inits), np.concatenate(labs), prob_col


def build_adversarial_inits(grid: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return stripes/checker/half-half with one-cell tipping, density nearly 0.5 but not exactly."""
    ii, jj = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    cases = []
    names = []
    labels = []

    def with_tip(s, target):
        if target == 1:
            # Add a 1 to make majority 1
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
        sh = with_tip(sh, target)
        cases.append(sh.copy()); names.append(f"stripes_h_lbl{target}"); labels.append(target)

        sv = np.zeros((grid, grid), dtype=np.uint8); sv[:, ::2] = 1
        sv = with_tip(sv, target)
        cases.append(sv.copy()); names.append(f"stripes_v_lbl{target}"); labels.append(target)

        cb = ((ii + jj) % 2).astype(np.uint8)
        cb = with_tip(cb, target)
        cases.append(cb.copy()); names.append(f"checker_lbl{target}"); labels.append(target)

        bc = ((ii // 2 + jj // 2) % 2).astype(np.uint8)
        bc = with_tip(bc, target)
        cases.append(bc.copy()); names.append(f"block_checker_lbl{target}"); labels.append(target)

        hh = np.full((grid, grid), target, dtype=np.uint8)
        hh[:, : grid // 2] = 1 - target
        hh = with_tip(hh, target)
        cases.append(hh.copy()); names.append(f"half_half_lbl{target}"); labels.append(target)

    return np.stack(cases), np.asarray(labels, dtype=np.uint8), names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sid", default="sid:58ed6b657afb")
    parser.add_argument("--amplifier", default="moore81")
    parser.add_argument("--p-F-values", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=1024)
    parser.add_argument("--final-det-steps", type=int, default=512,
                        help="Optional deterministic majority phase after stochastic phase.")
    parser.add_argument("--trials-per-side", type=int, default=64)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", choices=["mix", "mix_then_det"], default="mix_then_det")
    parser.add_argument("--test-adversarial", action="store_true",
                        help="Also test on adversarial structured inputs")
    parser.add_argument("--output", type=Path, default=Path("fates_results.json"))
    parser.add_argument("--backend", default="mlx")
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    F_bits = catalog.lut_bits[catalog.resolve_rule_ref(args.sid)].astype(np.uint8)
    backend = create_backend(args.backend)
    rng = np.random.default_rng(args.seed)

    # Random initial states
    random_inits, random_labels, _ = build_random_inits(rng, args.grid, args.probabilities, args.trials_per_side)
    if args.test_adversarial:
        adv_inits, adv_labels, adv_names = build_adversarial_inits(args.grid, rng)
    else:
        adv_inits = np.zeros((0, args.grid, args.grid), dtype=np.uint8)
        adv_labels = np.zeros((0,), dtype=np.uint8)
        adv_names = []

    runs = []
    for p_F in args.p_F_values:
        for name, inits, labs in [("random", random_inits, random_labels),
                                    ("adversarial", adv_inits, adv_labels)]:
            if inits.size == 0:
                continue
            t0 = time.time()
            # Use separate RNG stream per config so results are reproducible
            rng_stream = np.random.default_rng(args.seed + int(p_F * 10000) + len(inits))
            if args.mode == "mix":
                r = run_fates_mix(
                    F_bits, args.amplifier,
                    p_F=p_F, total_steps=args.total_steps,
                    initial_states=inits, labels=labs,
                    backend=backend, rng=rng_stream,
                )
            else:
                r = run_fates_mix_with_cutoff(
                    F_bits, args.amplifier,
                    p_F=p_F, max_steps=args.total_steps,
                    final_majority_steps=args.final_det_steps,
                    initial_states=inits, labels=labs,
                    backend=backend, rng=rng_stream,
                )
            r["p_F"] = p_F
            r["dataset"] = name
            r["grid"] = args.grid
            r["elapsed_seconds"] = time.time() - t0
            runs.append(r)
            print(
                f"p_F={p_F:.2f}  {name:<12} grid={args.grid:>4d} "
                f"cc={r['correct_consensus_rate']:.3f} "
                f"cons={r['consensus_rate']:.3f} "
                f"fma={r['final_majority_accuracy']:.3f} "
                f"[{r['elapsed_seconds']:.1f}s]"
            )

    args.output.write_text(json.dumps({
        "sid": args.sid,
        "amplifier": args.amplifier,
        "mode": args.mode,
        "total_steps": args.total_steps,
        "final_det_steps": args.final_det_steps,
        "grid": args.grid,
        "runs": runs,
        "adversarial_case_names": adv_names,
    }, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
