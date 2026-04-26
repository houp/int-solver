"""Hybrid stochastic + deterministic schedule:

Stage 1 (stochastic, T_mix steps):
    At each cell, with prob p_F apply preprocessor F (sid:58ed6).
    Otherwise apply "breaker" amplifier M_break (e.g. moore9 which flips stripes).

Stage 2 (deterministic, T_det steps):
    Apply the main amplifier M_main (e.g. moore81) to finalize.

Optional Stage 0 (deterministic pre-phase, T_pre_det steps):
    Apply F alone for T_pre_det steps before the mix. This gives the
    preprocessor its normal "canonicalization" action before the noise
    breaks any residual symmetries.

The stochastic stage ONLY exists to break translationally-symmetric fixed
points of the deterministic schedule. It does not need to be long, but
must be long enough to sufficiently randomize the state.
"""
from __future__ import annotations

import argparse
import itertools
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
        for name, template in [
            ("stripes_h", (lambda: (np.zeros((grid, grid), dtype=np.uint8)))),
            ("stripes_v", (lambda: (np.zeros((grid, grid), dtype=np.uint8)))),
            ("checker", (lambda: ((ii + jj) % 2).astype(np.uint8))),
            ("block_checker", (lambda: ((ii // 2 + jj // 2) % 2).astype(np.uint8))),
            ("half_half", (lambda: np.zeros((grid, grid), dtype=np.uint8))),
        ]:
            s = template()
            if name == "stripes_h":
                s[::2] = 1
            elif name == "stripes_v":
                s[:, ::2] = 1
            elif name == "half_half":
                s[:] = target
                s[:, : grid // 2] = 1 - target
            s = with_tip(s, target)
            cases.append(s.copy()); names.append(f"{name}_lbl{target}"); labels.append(target)
    return np.stack(cases), np.asarray(labels, dtype=np.uint8), names


def build_random_inits(rng, grid, probabilities, trials_per_side):
    inits, labs = [], []
    for p in probabilities:
        init = (rng.random((trials_per_side, grid, grid)) < p).astype(np.uint8)
        totals = init.reshape(trials_per_side, -1).sum(axis=1)
        lab = (totals > (grid * grid / 2)).astype(np.uint8)
        inits.append(init); labs.append(lab)
    return np.concatenate(inits), np.concatenate(labs)


def _pair_indices(states):
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


def apply_lut(states, tiled_bits):
    import mlx.core as mx
    indices = _pair_indices(states)
    out = mx.take_along_axis(tiled_bits[:, None, None, :], indices[..., None], axis=3)
    return mx.squeeze(out, axis=3).astype(mx.uint8)


def apply_breaker(states, kind: str, tiled_luts: dict):
    if kind in RADIUS_KINDS:
        return apply_radius_step_mlx(states, kind)
    return apply_lut(states, tiled_luts[kind])


def mix_step(states, tiled_F, breaker_kind, tiled_breaker_luts, p_F, rng):
    import mlx.core as mx
    F_out = apply_lut(states, tiled_F)
    M_out = apply_breaker(states, breaker_kind, tiled_breaker_luts)
    batch, H, W = states.shape
    mask_np = (rng.random((batch, H, W)) < p_F).astype(np.uint8)
    mask = mx.array(mask_np)
    return mx.where(mask == 1, F_out, M_out).astype(mx.uint8)


def run_hybrid(
    F_bits: np.ndarray,
    breaker_kind: str,
    main_kind: str,
    *,
    T_pre_det: int,
    T_mix: int,
    p_F: float,
    T_det: int,
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

    tiled_breaker_luts = {}
    if breaker_kind not in RADIUS_KINDS:
        bits = np.asarray(build_amplifier(breaker_kind), dtype=np.uint8)
        tiled_breaker_luts[breaker_kind] = backend.asarray(np.tile(bits, (batch, 1)), dtype="uint8")

    # Stage 0: deterministic F only
    for _ in range(T_pre_det):
        states = apply_lut(states, tiled_F)

    # Stage 1: stochastic mix
    for _ in range(T_mix):
        states = mix_step(states, tiled_F, breaker_kind, tiled_breaker_luts, p_F, rng)

    # Stage 2: deterministic main amplifier
    for _ in range(T_det):
        if main_kind in RADIUS_KINDS:
            states = apply_radius_step_mlx(states, main_kind)
        else:
            bits = np.asarray(build_amplifier(main_kind), dtype=np.uint8)
            tiled = backend.asarray(np.tile(bits, (batch, 1)), dtype="uint8")
            states = apply_lut(states, tiled)

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
        "final_density_mean": float(final.mean()),
        "num_failures": int((~correct).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sid", default="sid:58ed6b657afb")
    parser.add_argument("--breaker", default="moore9")
    parser.add_argument("--main", default="moore81")
    parser.add_argument("--T-pre-det", type=int, default=128)
    parser.add_argument("--T-mix", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--p-F", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--T-det", type=int, default=1024)
    parser.add_argument("--grids", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--trials-per-side", type=int, default=64)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-adversarial", action="store_true")
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("hybrid_stochastic.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    F_bits = catalog.lut_bits[catalog.resolve_rule_ref(args.sid)].astype(np.uint8)
    backend = create_backend(args.backend)

    runs = []
    for grid in args.grids:
        rng = np.random.default_rng(args.seed + grid)
        random_inits, random_labels = build_random_inits(rng, grid, args.probabilities, args.trials_per_side)
        if args.test_adversarial:
            adv_inits, adv_labels, adv_names = build_adversarial_inits(grid)
        else:
            adv_inits = np.zeros((0, grid, grid), dtype=np.uint8)
            adv_labels = np.zeros((0,), dtype=np.uint8)
            adv_names = []

        for T_mix, p_F in itertools.product(args.T_mix, args.p_F):
            for name, inits, labs in [("random", random_inits, random_labels),
                                        ("adversarial", adv_inits, adv_labels)]:
                if inits.size == 0:
                    continue
                rng_stream = np.random.default_rng(args.seed + int(p_F * 10000) + grid + T_mix * 7 + len(inits))
                t0 = time.time()
                r = run_hybrid(
                    F_bits, args.breaker, args.main,
                    T_pre_det=args.T_pre_det,
                    T_mix=T_mix, p_F=p_F,
                    T_det=args.T_det,
                    initial_states=inits, labels=labs,
                    rng=rng_stream, backend=backend,
                )
                r["dataset"] = name
                r["grid"] = grid
                r["T_mix"] = T_mix
                r["p_F"] = p_F
                r["elapsed_seconds"] = time.time() - t0
                runs.append(r)
                print(
                    f"grid={grid:>3d} T_mix={T_mix:>3d} p_F={p_F:.2f} {name:<12} "
                    f"cc={r['correct_consensus_rate']:.3f} "
                    f"fma={r['final_majority_accuracy']:.3f} "
                    f"fails={r['num_failures']:>3d}/{len(labs)}  [{r['elapsed_seconds']:.1f}s]"
                )

    args.output.write_text(json.dumps({
        "schedule": {
            "sid": args.sid,
            "breaker": args.breaker,
            "main": args.main,
            "T_pre_det": args.T_pre_det,
            "T_det": args.T_det,
        },
        "runs": runs,
    }, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
