"""Auto-tune the strict-CA mixture weights via random search over the
Dirichlet simplex.

Fixed rule set:
    R_0 = F_0 (sid:58ed6b657afb), the outer-monotone NCCA preprocessor;
    R_1..R_4 = traffic_{NE, NW, SE, SW};
    R_5 = M_{Moore9} (Moore-9 majority).

The search samples N_TRIALS weight vectors from a Dirichlet(alpha)
distribution, evaluates each on a shared test battery (random +
adversarial inputs at given (L, delta)), and records the
correct-consensus rates.

Composite score (higher = better):
    score(w) = min(random_cc, adversarial_cc)
             - lambda * (median_steps_to_consensus / max_steps)

The first term forces both random AND adversarial to be high; the
second term gently rewards faster convergence as a tie-breaker.

The output is persisted as a sortable list so the user can pick the
operating point most appropriate for their use case.
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
from strict_ca_classifier import resolve_rule_bits, run_strict_ca, _correct_consensus
from density_margin_sweep import (
    build_random_batch_at_density,
    build_adversarial_at_density,
)


def evaluate_mixture(
    rule_lut_list,
    weights,
    *,
    initial_states,
    labels,
    L: int,
    max_steps: int,
    backend,
    seed: int,
):
    rng = np.random.default_rng(seed)
    out = run_strict_ca(
        initial_states,
        rule_lut_list=rule_lut_list,
        probabilities=list(weights),
        max_steps=max_steps,
        until_consensus=True,
        backend=backend,
        rng=rng,
    )
    correct = _correct_consensus(out["final_totals"], labels, L * L)
    times = [t for t in out["time_to_consensus"] if t is not None]
    return {
        "correct_consensus_rate": float(correct.mean()),
        "n_correct": int(correct.sum()),
        "n_consensus": len(times),
        "n_trials": len(labels),
        "median_steps_to_consensus": (float(np.median(times)) if times else float("inf")),
        "max_steps_cap": max_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sid", default="sid:58ed6b657afb")
    parser.add_argument("--rules", nargs="+",
                        default=[
                            "sid:58ed6b657afb",
                            "traffic_ne", "traffic_nw",
                            "traffic_se", "traffic_sw",
                            "moore_maj",
                        ],
                        help="Rule set; mixture weights are searched over the simplex.")
    parser.add_argument("--N-trials", type=int, default=100,
                        help="Number of Dirichlet samples to evaluate")
    parser.add_argument("--alpha-base", type=float, default=1.0,
                        help="Dirichlet concentration; 1 = uniform on simplex")
    parser.add_argument("--alpha-near-baseline", type=float, default=0.0,
                        help="If > 0, blend the Dirichlet samples toward the published"
                             " baseline weights (0.70, 4*0.05, 0.10) using this scale."
                             " Useful for local search around the known-good point.")
    parser.add_argument("--L", type=int, default=96)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--n-random-per-side", type=int, default=16)
    parser.add_argument("--max-steps-mult", type=int, default=128)
    parser.add_argument("--lambda-time", type=float, default=0.05,
                        help="Coefficient for the time-cost penalty in the composite score")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path,
                        default=Path("results/density_classification/2026-04-30/schedule_autotune.json"))
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    catalog = load_binary_catalog(
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.bin"),
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.json"),
    )
    rule_lut_list = [resolve_rule_bits(name, catalog=catalog) for name in args.rules]
    K = len(rule_lut_list)
    backend = create_backend("mlx")

    # Build the test battery once
    rng_build = np.random.default_rng(args.seed * 7 + args.L)
    density_above = 0.5 + args.delta
    density_below = 0.5 - args.delta
    init_above, lab_above = build_random_batch_at_density(
        rng_build, args.L, density_above, args.n_random_per_side)
    init_below, lab_below = build_random_batch_at_density(
        rng_build, args.L, density_below, args.n_random_per_side)
    random_init = np.concatenate([init_above, init_below])
    random_labels = np.concatenate([lab_above, lab_below])
    adv_init, adv_labels, adv_names = build_adversarial_at_density(
        args.L, density_above, rng_build,
    )

    print(f"Rule set ({K} rules):")
    for r in args.rules:
        print(f"  - {r}")
    print(f"Test battery: {len(random_labels)} random + {len(adv_labels)} adversarial trials at L={args.L}, delta={args.delta}")
    print()

    # Baseline weights for reference (the published mixture)
    baseline_weights = np.array([0.70, 0.05, 0.05, 0.05, 0.05, 0.10])
    if K != 6 or not np.isclose(baseline_weights.sum(), 1.0):
        baseline_weights = np.full(K, 1.0 / K)
    print(f"Baseline weights: {baseline_weights}")

    rng_sample = np.random.default_rng(args.seed)
    max_steps = args.max_steps_mult * args.L

    results = []
    t0 = time.time()
    for trial in range(args.N_trials):
        # Sample weight vector
        if trial == 0:
            # First trial: evaluate the baseline exactly
            w = baseline_weights.copy()
            label = "baseline"
        elif args.alpha_near_baseline > 0:
            # Local search: bias toward baseline with concentration alpha_near_baseline
            alpha_vec = baseline_weights * args.alpha_near_baseline + 1.0
            w = rng_sample.dirichlet(alpha_vec)
            label = "near_baseline"
        else:
            # Uniform on simplex (Dirichlet with alpha = alpha_base for all components)
            alpha_vec = np.full(K, args.alpha_base)
            w = rng_sample.dirichlet(alpha_vec)
            label = "uniform"

        # Evaluate on random + adversarial
        t_eval0 = time.time()
        random_res = evaluate_mixture(
            rule_lut_list, w,
            initial_states=random_init, labels=random_labels,
            L=args.L, max_steps=max_steps,
            backend=backend, seed=args.seed + trial * 7,
        )
        adv_res = evaluate_mixture(
            rule_lut_list, w,
            initial_states=adv_init, labels=adv_labels,
            L=args.L, max_steps=max_steps,
            backend=backend, seed=args.seed + trial * 11,
        )
        elapsed_eval = time.time() - t_eval0

        # Composite score
        rand_cc = random_res["correct_consensus_rate"]
        adv_cc = adv_res["correct_consensus_rate"]
        med_steps = random_res["median_steps_to_consensus"]
        time_cost = (med_steps / max_steps) if med_steps != float("inf") else 1.0
        score = min(rand_cc, adv_cc) - args.lambda_time * time_cost

        row = {
            "trial": trial,
            "label": label,
            "weights": w.tolist(),
            "random_cc": rand_cc,
            "adversarial_cc": adv_cc,
            "score": score,
            "median_steps_random": med_steps,
            "median_steps_adv": adv_res["median_steps_to_consensus"],
            "elapsed_eval_seconds": elapsed_eval,
        }
        results.append(row)

        # Progress line
        elapsed_total = time.time() - t0
        weights_str = "[" + ", ".join(f"{x:.3f}" for x in w) + "]"
        print(f"  [{trial + 1:>3}/{args.N_trials}] "
              f"score={score:.4f} "
              f"rand={rand_cc:.3f} adv={adv_cc:.3f} "
              f"med_steps={med_steps:.0f} "
              f"w={weights_str} "
              f"({elapsed_eval:.1f}s/eval, {elapsed_total:.1f}s total)")

    # Sort by score descending
    results.sort(key=lambda r: -r["score"])
    print()
    print(f"=== Top 10 mixtures by composite score ===")
    for i, r in enumerate(results[:10]):
        weights_str = "[" + ", ".join(f"{x:.3f}" for x in r["weights"]) + "]"
        print(f"  {i:>2}: score={r['score']:.4f} "
              f"rand={r['random_cc']:.3f} adv={r['adversarial_cc']:.3f} "
              f"med={r['median_steps_random']:.0f} "
              f"w={weights_str}"
              f"  ({r['label']})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "args": vars(args) | {
            "output": str(args.output),
            "baseline_weights": baseline_weights.tolist(),
            "rules": list(args.rules),
        },
        "results_sorted_by_score": results,
    }, indent=2, default=str))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
