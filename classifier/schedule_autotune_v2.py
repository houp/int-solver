"""Auto-tune the strict-CA mixture weights against the parameterized
adversarial battery and the random-input objective, using the
label-symmetric ``min(cc_0, cc_1)`` metric uncovered in tier-3.

This is a successor to ``schedule_autotune.py``.  The differences:

1. The adversarial component of the score is the full parameterized
   battery (~444 cases per density set; see
   ``parameterized_adversarial.py``) rather than the saturated
   eight-family static battery.
2. The score is computed per (density, label) cell and aggregated as
   ``min`` over labels (so a 0-bias is penalized hard), then
   ``mean`` over densities.
3. Random and adversarial scores are combined as the unweighted
   minimum, which forces both to be high at the optimum.
4. We additionally keep the random-cc and adversarial-cc components
   in the result row so post-hoc Pareto analysis is possible.

Compute-budget shape (default at ``L=64``):
  - 50 weight-vector trials
  - per trial: 2 * n_random_per_side random inputs (default 16) +
    full parameterized battery (~444 cases) at densities
    {0.51, 0.52, 0.55} and complements
  - ~115 s/trial on M2 Max MLX -> ~95 minutes total wall.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
from strict_ca_classifier import resolve_rule_bits, run_strict_ca, _correct_consensus
from density_margin_sweep import build_random_batch_at_density
from parameterized_adversarial import build_parameterized_battery


def label_symmetric_cc(correct: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Return (cc_label0, cc_label1, min(cc_label0, cc_label1))."""
    correct = np.asarray(correct, dtype=bool)
    labels = np.asarray(labels)
    m0 = labels == 0
    m1 = labels == 1
    cc0 = float(correct[m0].mean()) if m0.any() else 1.0
    cc1 = float(correct[m1].mean()) if m1.any() else 1.0
    return cc0, cc1, min(cc0, cc1)


def run_one(rule_lut_list, weights, *, inits, labels, max_steps, backend, seed):
    rng = np.random.default_rng(seed)
    out = run_strict_ca(
        inits, rule_lut_list=rule_lut_list, probabilities=list(weights),
        max_steps=max_steps, until_consensus=True,
        backend=backend, rng=rng,
    )
    L = inits.shape[-1]
    correct = _correct_consensus(out["final_totals"], labels, L * L)
    times = [t for t in out["time_to_consensus"] if t is not None]
    return correct, (float(np.median(times)) if times else float(max_steps))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", nargs="+",
                        default=[
                            "sid:58ed6b657afb",
                            "traffic_ne", "traffic_nw",
                            "traffic_se", "traffic_sw",
                            "moore_maj",
                        ])
    parser.add_argument("--N-trials", type=int, default=50)
    parser.add_argument("--reference-weights", type=float, nargs="+", default=None,
                        help="Reference weights for trial 0 and the alpha_near_rebalanced "
                             "Dirichlet anchor. Defaults to the 6-rule rebalanced weights "
                             "if K=6, else uniform(1/K).")
    parser.add_argument("--alpha-base", type=float, default=1.0)
    parser.add_argument("--alpha-near-rebalanced", type=float, default=0.0,
                        help="If > 0, blend Dirichlet samples toward the autotune-rebalanced"
                             " weights using this concentration. 0 = pure simplex sampling.")
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--rand-densities", type=float, nargs="+",
                        default=[0.51, 0.52, 0.55])
    parser.add_argument("--adv-densities", type=float, nargs="+",
                        default=[0.51, 0.52, 0.55])
    parser.add_argument("--n-random-per-side", type=int, default=16)
    parser.add_argument("--max-steps-mult", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path,
                        default=Path("results/density_classification/2026-05-01/schedule_autotune_v2.json"))
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    catalog = load_binary_catalog(
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.bin"),
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.json"),
    )
    rule_lut_list = [resolve_rule_bits(name, catalog=catalog) for name in args.rules]
    K = len(rule_lut_list)
    backend = create_backend("mlx")

    # Build random batch (both labels for every density)
    rng_build = np.random.default_rng(args.seed * 7 + args.L)
    rand_inits = []
    rand_labels = []
    for d in args.rand_densities:
        for actual in (d, 1.0 - d):
            inits, lab = build_random_batch_at_density(
                rng_build, args.L, actual, args.n_random_per_side)
            rand_inits.append(inits); rand_labels.append(lab)
    rand_inits = np.concatenate(rand_inits)
    rand_labels = np.concatenate(rand_labels)

    # Build parameterized adversarial battery
    rng_adv = np.random.default_rng(args.seed * 7919)
    adv_inits, adv_labels, adv_names = build_parameterized_battery(
        args.L, args.adv_densities, rng_adv)

    print(f"Rule set ({K} rules): {args.rules}")
    print(f"Random batch: {len(rand_labels)} trials at L={args.L}, "
          f"densities={args.rand_densities}")
    print(f"Adversarial battery: {len(adv_labels)} cases at L={args.L}, "
          f"densities={args.adv_densities}")
    print()

    # Reference weights (the anchor of the local search)
    if args.reference_weights is not None:
        rebalanced = np.array(args.reference_weights, dtype=float)
        if rebalanced.size != K:
            raise SystemExit(f"--reference-weights has {rebalanced.size} entries but K={K}")
        s = rebalanced.sum()
        if not np.isclose(s, 1.0):
            print(f"  (renormalising reference weights from sum {s:.6f})")
            rebalanced = rebalanced / s
    else:
        rebalanced = np.array([0.041, 0.049, 0.126, 0.017, 0.579, 0.188])
        if K != 6 or not np.isclose(rebalanced.sum(), 1.0):
            rebalanced = np.full(K, 1.0 / K)

    rng_sample = np.random.default_rng(args.seed)
    max_steps = args.max_steps_mult * args.L

    results = []
    t0 = time.time()
    for trial in range(args.N_trials):
        if trial == 0:
            w = rebalanced.copy(); label = "rebalanced"
        elif args.alpha_near_rebalanced > 0:
            w = rng_sample.dirichlet(rebalanced * args.alpha_near_rebalanced + 1.0)
            label = "near_rebalanced"
        else:
            w = rng_sample.dirichlet(np.full(K, args.alpha_base))
            label = "uniform"

        # Random-input score
        t_eval0 = time.time()
        correct_r, med_r = run_one(
            rule_lut_list, w, inits=rand_inits, labels=rand_labels,
            max_steps=max_steps, backend=backend, seed=args.seed + trial * 7)
        rand_cc0, rand_cc1, rand_min = label_symmetric_cc(correct_r, rand_labels)

        # Adversarial-input score
        correct_a, med_a = run_one(
            rule_lut_list, w, inits=adv_inits, labels=adv_labels,
            max_steps=max_steps, backend=backend, seed=args.seed + trial * 11)
        adv_cc0, adv_cc1, adv_min = label_symmetric_cc(correct_a, adv_labels)

        # Combined symmetric score: min over (random_min, adv_min)
        score = min(rand_min, adv_min)

        # Per-family adversarial breakdown (top-level family only)
        fam_cc = defaultdict(lambda: [0, 0])
        for nm, c in zip(adv_names, correct_a):
            base = nm.rsplit("_d", 1)[0]
            for sep in ["_p", "_b", "_w", "_n", "_kx", "_e"]:
                base = base.split(sep)[0]
            base = base.rstrip("_")
            fam_cc[base][0] += int(c); fam_cc[base][1] += 1
        fam_breakdown = {k: v[0] / v[1] for k, v in fam_cc.items()}

        elapsed_eval = time.time() - t_eval0
        row = {
            "trial": trial, "label": label,
            "weights": w.tolist(),
            "random_cc0": rand_cc0, "random_cc1": rand_cc1,
            "random_min": rand_min,
            "adv_cc0": adv_cc0, "adv_cc1": adv_cc1, "adv_min": adv_min,
            "score": score,
            "med_steps_random": med_r, "med_steps_adv": med_a,
            "fam_cc": fam_breakdown,
            "elapsed_eval_seconds": elapsed_eval,
        }
        results.append(row)

        weights_str = "[" + ", ".join(f"{x:.3f}" for x in w) + "]"
        elapsed_total = time.time() - t0
        print(f"  [{trial + 1:>3}/{args.N_trials}] "
              f"score={score:.4f} "
              f"rand_min={rand_min:.3f} (cc0={rand_cc0:.3f}, cc1={rand_cc1:.3f})  "
              f"adv_min={adv_min:.3f} (cc0={adv_cc0:.3f}, cc1={adv_cc1:.3f})  "
              f"med_r={med_r:.0f}  w={weights_str}  ({elapsed_eval:.1f}s; total {elapsed_total:.0f}s)")

    results.sort(key=lambda r: -r["score"])
    print()
    print("=== Top 10 mixtures by symmetric score ===")
    for i, r in enumerate(results[:10]):
        weights_str = "[" + ", ".join(f"{x:.3f}" for x in r["weights"]) + "]"
        print(f"  {i:>2}: score={r['score']:.4f}  rand_min={r['random_min']:.3f}  "
              f"adv_min={r['adv_min']:.3f}  med={r['med_steps_random']:.0f}  "
              f"w={weights_str}  ({r['label']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "args": vars(args) | {
            "output": str(args.output),
            "rebalanced_reference_weights": rebalanced.tolist(),
            "rules": list(args.rules),
        },
        "results_sorted_by_score": results,
    }, indent=2, default=str))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
