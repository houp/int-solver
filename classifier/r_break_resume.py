"""Stage-1+2 only R_break search, designed to fit a ~20-30 min budget
with visible incremental progress.

Differences from r_break_search.py:
- forces line-buffered stdout (so users can `tail -f` the log);
- skips the stage-3 multi-seed validation (run separately afterward
  with classifier/strict_ca_validate.py on the top-K rules);
- prints one progress line per rule in stage 2.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# --- repo-root path bootstrap ---
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
_CATALOG_DIR = _REPO_ROOT / "catalogs"
# --------------------------------

from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend
from witek_sampler import sample_random_lut_batch
from r_break_probe import probe_rules
from r_break_search import stage2_full_eval_one_rule
from strict_ca_classifier import resolve_rule_bits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sid", default="sid:58ed6b657afb")
    parser.add_argument("--K-sample", type=int, default=20_000)
    parser.add_argument("--n-top-stage2", type=int, default=60,
                        help="Top-N rules from stage 1 to evaluate fully")
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--trials-per-side", type=int, default=8)
    parser.add_argument("--max-steps-mult", type=int, default=64)
    parser.add_argument("--p-majority-probe", type=float, default=0.05)
    parser.add_argument("--T-probe", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path,
                        default=Path("results/density_classification/2026-04-30/r_break_resume.json"))
    args = parser.parse_args()

    # Force line-buffered stdout for visible incremental progress
    sys.stdout.reconfigure(line_buffering=True)

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    print(f"[stage 1] sampling {args.K_sample} rules from witek/ ...")
    t0 = time.time()
    luts, gidx = sample_random_lut_batch(
        args.K_sample, seed=args.seed, return_global_indices=True,
    )
    t_sample = time.time() - t0
    print(f"[stage 1] sampled in {t_sample:.1f}s; LUTs shape={luts.shape}")

    print(f"[stage 1] probing (L={args.L}, T={args.T_probe}, p_M={args.p_majority_probe}) ...")
    t0 = time.time()
    rng_probe = np.random.default_rng(args.seed * 17)
    probe_out = probe_rules(
        luts, L=args.L, T_probe=args.T_probe,
        p_majority=args.p_majority_probe, rng=rng_probe,
    )
    t_probe = time.time() - t0
    print(f"[stage 1] probe done in {t_probe:.1f}s "
          f"({args.K_sample / max(t_probe, 1e-6):.0f} rules/s); "
          f"score min={probe_out['combined'].min():.3f}, "
          f"med={np.median(probe_out['combined']):.3f}, "
          f"max={probe_out['combined'].max():.3f}")

    order = np.argsort(probe_out["combined"])
    top_indices = order[:args.n_top_stage2]

    # ------------------------------------------------------------------
    # Resolve baseline F + traffic + majority for stage 2
    # ------------------------------------------------------------------
    catalog = load_binary_catalog(
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.bin"),
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.json"),
    )
    F_bits = np.asarray(
        resolve_rule_bits(args.baseline_sid, catalog=catalog),
        dtype=np.uint8,
    )
    traffic_diag_luts = {
        name: resolve_rule_bits(name)
        for name in ["traffic_ne", "traffic_nw", "traffic_se", "traffic_sw"]
    }
    moore_lut = resolve_rule_bits("moore_maj")
    backend = create_backend("mlx")

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    print(f"[stage 2] evaluating top {len(top_indices)} candidates "
          f"at L={args.L}, delta={args.delta}, trials/side={args.trials_per_side}, "
          f"max_steps_mult={args.max_steps_mult} ...")
    t0 = time.time()
    stage2_results = []
    for i, j in enumerate(top_indices):
        t_eval0 = time.time()
        per = stage2_full_eval_one_rule(
            luts[j],
            F_bits=F_bits,
            traffic_diag_luts=traffic_diag_luts,
            moore_lut=moore_lut,
            L=args.L,
            delta=args.delta,
            n_random_per_side=args.trials_per_side,
            max_steps_mult=args.max_steps_mult,
            seed=args.seed + i,
            backend=backend,
        )
        elapsed_eval = time.time() - t_eval0
        elapsed_total = time.time() - t0
        stage2_results.append({
            "rank_in_probe": int(i),
            "global_index": int(gidx[j]),
            "stage1_score": float(probe_out["combined"][j]),
            **per,
            "lut_hex": "".join(f"{b:01x}" for b in luts[j]),
        })
        # One progress line per rule
        print(f"  [{i + 1:>3}/{len(top_indices)}]"
              f" idx={int(gidx[j]):>11}"
              f" stage1={probe_out['combined'][j]:.3f}"
              f" adv_cc={per['adversarial_cc']:.3f}"
              f" rand_cc={per['random_cc']:.3f}"
              f" rand_med={per['random_median_steps']:.0f}"
              f" elapsed_eval={elapsed_eval:.1f}s"
              f" total={elapsed_total:.1f}s")

    # Sort by adversarial cc (desc), then random cc (desc), then median steps
    stage2_results.sort(
        key=lambda r: (-r["adversarial_cc"], -r["random_cc"], r["random_median_steps"])
    )

    print()
    print(f"=== Stage 2 top 20 (sorted by adversarial cc, then random cc) ===")
    for i, r in enumerate(stage2_results[:20]):
        print(f"  {i:>2}: idx={r['global_index']:>11}  "
              f"adv_cc={r['adversarial_cc']:.3f} ({r['adversarial_consensus']}/{r['adversarial_total']})  "
              f"rand_cc={r['random_cc']:.3f} ({r['random_consensus']}/{r['random_total']})  "
              f"rand_med={r['random_median_steps']:.0f}  "
              f"stage1={r['stage1_score']:.3f}")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "args": vars(args) | {
            "output": str(args.output),
            "baseline_sid": args.baseline_sid,
        },
        "elapsed": {"sample": t_sample, "probe": t_probe},
        "stage1_score_distribution": {
            "min": float(probe_out["combined"].min()),
            "p25": float(np.percentile(probe_out["combined"], 25)),
            "p50": float(np.percentile(probe_out["combined"], 50)),
            "p75": float(np.percentile(probe_out["combined"], 75)),
            "max": float(probe_out["combined"].max()),
        },
        "stage2_results": stage2_results,
    }, indent=2, default=str))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
