"""End-to-end R_break search pipeline.

Stages:
1. Sample K rules from the witek/ corpus.
2. Cheap GPU-batched stripe-disruption probe (stochastic mixture).
3. Pick top-N1 by combined score; evaluate each in the full strict-CA
   classifier as a 7th component (replacing weight 0.05 from the
   majority slot).  Single-seed evaluation at L=64 with random +
   adversarial inputs.
4. Pick top-N2 from stage 3 by adversarial cc; multi-seed validation
   at L in {64, 96, 128}.
5. Persist the winning R_break rule(s) and full evaluation.

Each stage filters down by ~10x.  Stage 1: 50k -> 200; Stage 3: 200 ->
~10; Stage 4: ~10 -> top 1-3.
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
from witek_sampler import sample_random_lut_batch, WitekIndex
from r_break_probe import probe_rules
from strict_ca_classifier import resolve_rule_bits, run_strict_ca, _correct_consensus
from density_margin_sweep import (
    build_random_batch_at_density,
    build_adversarial_at_density,
)


def stage1_sample_and_probe(
    K_sample: int,
    *,
    L_probe: int,
    T_probe: int,
    p_majority: float,
    seed: int,
    n_top_persist: int,
) -> dict:
    """Sample, probe, and return top-N candidate LUTs + scores."""
    print(f"[stage 1] sampling {K_sample} rules ...")
    t0 = time.time()
    luts, gidx = sample_random_lut_batch(K_sample, seed=seed, return_global_indices=True)
    t_sample = time.time() - t0

    print(f"[stage 1] probing (L={L_probe}, T={T_probe}, p_M={p_majority}) ...")
    t0 = time.time()
    rng = np.random.default_rng(seed * 17)
    out = probe_rules(luts, L=L_probe, T_probe=T_probe, p_majority=p_majority, rng=rng)
    t_probe = time.time() - t0

    combined = out["combined"]
    order = np.argsort(combined)
    top = order[:n_top_persist]
    return {
        "elapsed_sample_seconds": t_sample,
        "elapsed_probe_seconds": t_probe,
        "top_indices": gidx[top].tolist(),
        "top_luts": luts[top],
        "top_scores": combined[top].tolist(),
        "top_diagnostics": [
            {
                "stripes_h_score": float(out["stripes_h_score"][j]),
                "stripes_v_score": float(out["stripes_v_score"][j]),
                "block_checker_score": float(out["block_checker_score"][j]),
                "stripes_h_h_disagree": float(out["stripes_h_h_disagree"][j]),
                "stripes_h_v_disagree": float(out["stripes_h_v_disagree"][j]),
                "stripes_h_density": float(out["stripes_h_density"][j]),
            }
            for j in top
        ],
        "score_full_distribution": {
            "min": float(combined.min()),
            "p25": float(np.percentile(combined, 25)),
            "p50": float(np.percentile(combined, 50)),
            "p75": float(np.percentile(combined, 75)),
            "max": float(combined.max()),
        },
    }


def stage2_full_eval_one_rule(
    rule_lut: np.ndarray,
    *,
    F_bits: np.ndarray,
    traffic_diag_luts: dict[str, list[int]],
    moore_lut: list[int],
    L: int,
    delta: float,
    n_random_per_side: int,
    max_steps_mult: int,
    seed: int,
    backend,
) -> dict:
    """Insert rule_lut as R_break (weight 0.05) into the strict-CA mixture
    and evaluate at one (L, delta).
    Returns dict with random + adversarial cc, n_consensus, etc.

    The mixture: F=0.65, 4*diag traffic at 0.05, M_Moore9=0.10, R_break=0.05.
    """
    # Build mixture LUTs.
    rule_lut_list = [
        F_bits.tolist() if isinstance(F_bits, np.ndarray) else F_bits,
        traffic_diag_luts["traffic_ne"],
        traffic_diag_luts["traffic_nw"],
        traffic_diag_luts["traffic_se"],
        traffic_diag_luts["traffic_sw"],
        moore_lut,
        rule_lut.tolist() if isinstance(rule_lut, np.ndarray) else rule_lut,
    ]
    probabilities = [0.65, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05]

    rng_build = np.random.default_rng(seed + L * 1000 + int(delta * 1e6))
    density_above = 0.5 + delta
    density_below = 0.5 - delta
    init_above, lab_above = build_random_batch_at_density(
        rng_build, L, density_above, n_random_per_side)
    init_below, lab_below = build_random_batch_at_density(
        rng_build, L, density_below, n_random_per_side)
    random_init = np.concatenate([init_above, init_below])
    random_labels = np.concatenate([lab_above, lab_below])

    adv_init, adv_labels, adv_names = build_adversarial_at_density(
        L, density_above, rng_build)

    out_random = _eval_strict_ca(
        rule_lut_list, probabilities,
        random_init, random_labels, L,
        max_steps=max_steps_mult * L, seed=seed * 7 + L, backend=backend,
    )
    out_adv = _eval_strict_ca(
        rule_lut_list, probabilities,
        adv_init, adv_labels, L,
        max_steps=max_steps_mult * L, seed=seed * 11 + L, backend=backend,
    )
    return {
        "random_cc": out_random["correct_consensus_rate"],
        "random_consensus": out_random["n_consensus"],
        "random_total": out_random["n_trials"],
        "random_median_steps": out_random["median_steps_to_consensus"],
        "adversarial_cc": out_adv["correct_consensus_rate"],
        "adversarial_consensus": out_adv["n_consensus"],
        "adversarial_total": out_adv["n_trials"],
        "adversarial_median_steps": out_adv["median_steps_to_consensus"],
        "adversarial_per_case": [
            {"name": n, "label": int(adv_labels[i]),
             "correct": bool(out_adv["per_trial_correct"][i])}
            for i, n in enumerate(adv_names)
        ],
    }


def _eval_strict_ca(rule_lut_list, probabilities, init, labels, L,
                     *, max_steps, seed, backend):
    rng_run = np.random.default_rng(seed)
    out = run_strict_ca(
        init, rule_lut_list=rule_lut_list, probabilities=probabilities,
        max_steps=max_steps, until_consensus=True,
        backend=backend, rng=rng_run,
    )
    L2 = L * L
    correct = _correct_consensus(out["final_totals"], labels, L2)
    times = [t for t in out["time_to_consensus"] if t is not None]
    return {
        "correct_consensus_rate": float(correct.mean()),
        "n_correct": int(correct.sum()),
        "n_consensus": len(times),
        "n_trials": len(labels),
        "median_steps_to_consensus": float(np.median(times)) if times else float("inf"),
        "per_trial_correct": correct.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sid", default="sid:58ed6b657afb")
    parser.add_argument("--K-sample", type=int, default=50_000,
                        help="Stage 1: number of random rules from witek")
    parser.add_argument("--L-probe", type=int, default=64)
    parser.add_argument("--T-probe", type=int, default=64)
    parser.add_argument("--p-majority", type=float, default=0.05)
    parser.add_argument("--n-top-stage2", type=int, default=200,
                        help="Top-N rules from stage 1 to evaluate in stage 2 (full classifier)")
    parser.add_argument("--n-top-stage3", type=int, default=10,
                        help="Top-N rules from stage 2 to multi-seed validate in stage 3")

    parser.add_argument("--stage2-L", type=int, default=64)
    parser.add_argument("--stage2-delta", type=float, default=0.02)
    parser.add_argument("--stage2-trials", type=int, default=16)
    parser.add_argument("--stage2-max-mult", type=int, default=128)

    parser.add_argument("--stage3-Ls", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--stage3-deltas", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--stage3-trials", type=int, default=32)
    parser.add_argument("--stage3-max-mult", type=int, default=256)
    parser.add_argument("--stage3-seeds", type=int, nargs="+", default=[2026, 2027, 2028])

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path,
                        default=Path("results/density_classification/2026-04-29/r_break_search.json"))
    args = parser.parse_args()

    backend = create_backend("mlx")

    # Stage 1
    stage1 = stage1_sample_and_probe(
        K_sample=args.K_sample, L_probe=args.L_probe, T_probe=args.T_probe,
        p_majority=args.p_majority, seed=args.seed,
        n_top_persist=args.n_top_stage2,
    )
    print(f"[stage 1] done in "
          f"{stage1['elapsed_sample_seconds']:.1f}s sample + "
          f"{stage1['elapsed_probe_seconds']:.1f}s probe; "
          f"score min={stage1['score_full_distribution']['min']:.3f}, "
          f"keeping top {len(stage1['top_indices'])}")

    # Resolve baseline F + auxiliary rules.
    from ca_search.binary_catalog import load_binary_catalog
    catalog = load_binary_catalog(
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.bin"),
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.json"))
    F_bits = np.asarray(resolve_rule_bits(args.baseline_sid, catalog=catalog),
                         dtype=np.uint8)
    traffic_diag_luts = {
        name: resolve_rule_bits(name) for name in
        ["traffic_ne", "traffic_nw", "traffic_se", "traffic_sw"]
    }
    moore_lut = resolve_rule_bits("moore_maj")

    # Stage 2: full strict-CA evaluation per candidate
    print(f"[stage 2] evaluating top {len(stage1['top_indices'])} candidates"
          f" at L={args.stage2_L}, delta={args.stage2_delta} ...")
    t0 = time.time()
    stage2_results = []
    for i, (gidx, lut, score) in enumerate(zip(
        stage1["top_indices"], stage1["top_luts"], stage1["top_scores"]
    )):
        per = stage2_full_eval_one_rule(
            lut, F_bits=F_bits, traffic_diag_luts=traffic_diag_luts,
            moore_lut=moore_lut, L=args.stage2_L, delta=args.stage2_delta,
            n_random_per_side=args.stage2_trials,
            max_steps_mult=args.stage2_max_mult,
            seed=args.seed + i, backend=backend,
        )
        stage2_results.append({
            "global_index": int(gidx),
            "stage1_score": float(score),
            **per,
        })
        if (i + 1) % 10 == 0 or i + 1 == len(stage1["top_indices"]):
            elapsed = time.time() - t0
            print(f"  [stage 2] {i + 1}/{len(stage1['top_indices'])}  "
                  f"({elapsed:.1f}s elapsed, "
                  f"~{elapsed / (i + 1):.1f}s/rule)")

    # Pick top by adversarial cc; tie-break by random cc and (descending) speed.
    stage2_results.sort(
        key=lambda r: (-r["adversarial_cc"], -r["random_cc"], r["random_median_steps"])
    )

    print()
    print(f"[stage 2] top {min(20, len(stage2_results))} by adversarial cc:")
    for i, r in enumerate(stage2_results[:20]):
        print(f"  {i:>2}: idx={r['global_index']:>11}  "
              f"adv_cc={r['adversarial_cc']:.3f} "
              f"random_cc={r['random_cc']:.3f} "
              f"med_steps={r['random_median_steps']:.0f}")

    # Stage 3: multi-seed validation of top n_top_stage3 candidates
    top3 = stage2_results[:args.n_top_stage3]
    print()
    print(f"[stage 3] multi-seed validation of top {len(top3)} candidates ...")
    t0 = time.time()
    stage3_results = []
    for i, r in enumerate(top3):
        gidx = r["global_index"]
        # Re-fetch the LUT
        lut = stage1["top_luts"][stage1["top_indices"].index(gidx)]
        per_seed = []
        for seed_v in args.stage3_seeds:
            per_grid = []
            for L in args.stage3_Ls:
                for delta in args.stage3_deltas:
                    res = stage2_full_eval_one_rule(
                        lut, F_bits=F_bits, traffic_diag_luts=traffic_diag_luts,
                        moore_lut=moore_lut, L=L, delta=delta,
                        n_random_per_side=args.stage3_trials,
                        max_steps_mult=args.stage3_max_mult,
                        seed=seed_v + i, backend=backend,
                    )
                    per_grid.append({
                        "L": L, "delta": delta,
                        "random_cc": res["random_cc"],
                        "adversarial_cc": res["adversarial_cc"],
                        "random_median_steps": res["random_median_steps"],
                        "adversarial_median_steps": res["adversarial_median_steps"],
                    })
            per_seed.append({"seed": seed_v, "per_grid": per_grid})
        stage3_results.append({
            "global_index": gidx,
            "stage1_score": r["stage1_score"],
            "stage2_adv_cc": r["adversarial_cc"],
            "stage2_random_cc": r["random_cc"],
            "per_seed": per_seed,
        })
        elapsed = time.time() - t0
        print(f"  [stage 3] {i + 1}/{len(top3)} done ({elapsed:.1f}s)")

    summary = {
        "args": {
            "K_sample": args.K_sample,
            "n_top_stage2": args.n_top_stage2,
            "n_top_stage3": args.n_top_stage3,
            "stage2_L": args.stage2_L,
            "stage2_delta": args.stage2_delta,
            "stage3_Ls": args.stage3_Ls,
            "stage3_deltas": args.stage3_deltas,
        },
        "stage1": {
            "elapsed_sample_seconds": stage1["elapsed_sample_seconds"],
            "elapsed_probe_seconds": stage1["elapsed_probe_seconds"],
            "n_top_persisted": len(stage1["top_indices"]),
            "score_distribution": stage1["score_full_distribution"],
        },
        "stage2_results": stage2_results,
        "stage3_results": stage3_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
