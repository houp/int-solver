"""Paired head-to-head benchmark for strict-CA mixture candidates.

For each (L, delta) cell, run a fixed shared batch of initial
configurations (random + extended-adversarial) under each candidate
mixture across multiple seeds.  Report:

- per-candidate cc estimate with binomial 95% CI;
- pairwise McNemar-test results (for each (cand_a, cand_b), how many
  trials does only cand_a get right vs only cand_b, and the resulting
  two-sided p-value).

The pairing -- same initial configs across all candidates -- gives
much higher statistical power per trial than independent samples.
The dynamics are still stochastic per candidate (different per-cell
rule draws) so cc differences reflect mixture quality, not seed luck.

Candidates are specified as JSON dicts:
    {"name": "baseline", "rules": [...], "weights": [...]}

The list is loaded from a JSON file via --candidates-file.
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
from witek_sampler import WitekIndex, _read_offsets_from_file, _decode_records_to_luts
from extended_adversarial import build_extended_adversarial_inits


def _build_random_batch(L, density, n_per_side, rng):
    """Return (init, labels) for n_per_side trials at each of (0.5+d, 0.5-d)."""
    inits = []; labs = []
    for d in [density, -density]:
        rho = 0.5 + d
        n_total = L * L
        n_ones = int(round(rho * n_total))
        for _ in range(n_per_side):
            flat = np.zeros(n_total, dtype=np.uint8); flat[:n_ones] = 1
            rng.shuffle(flat)
            inits.append(flat.reshape(L, L))
            labs.append(1 if rho > 0.5 else 0)
    return np.stack(inits), np.asarray(labs, dtype=np.uint8)


def fetch_lut_by_name(name: str, *, catalog, witek_index: WitekIndex | None) -> list[int]:
    """Resolve a rule reference: built-in name, sid, legacy:N, rank:N, or witek:N."""
    if name.startswith("witek:"):
        gi = int(name.split(":", 1)[1])
        if witek_index is None:
            raise RuntimeError("witek index not loaded but witek: rule requested")
        path, byte_offset = witek_index.locate(gi)
        rec = _read_offsets_from_file(path, np.array([byte_offset], dtype=np.int64))
        return list(_decode_records_to_luts(rec)[0])
    return resolve_rule_bits(name, catalog=catalog)


def evaluate_candidate(
    rule_lut_list, weights,
    *, init, labels, L, max_steps, backend, seed,
):
    rng = np.random.default_rng(seed)
    out = run_strict_ca(
        init, rule_lut_list=rule_lut_list, probabilities=list(weights),
        max_steps=max_steps, until_consensus=True,
        backend=backend, rng=rng,
    )
    correct = _correct_consensus(out["final_totals"], labels, L * L)
    times = [t for t in out["time_to_consensus"] if t is not None]
    return {
        "correct_per_trial": correct.tolist(),
        "n_correct": int(correct.sum()),
        "n_trials": len(labels),
        "median_steps": float(np.median(times)) if times else float("inf"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-file", type=Path, required=True,
                        help="JSON file with a list of {name, rules, weights} dicts.")
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--n-seeds", type=int, default=16)
    parser.add_argument("--n-random-per-side", type=int, default=128)
    parser.add_argument("--adv-densities", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--max-steps-mult", type=int, default=192)
    parser.add_argument("--seed-base", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    catalog = load_binary_catalog(
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.bin"),
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.json"),
    )
    candidates = json.loads(args.candidates_file.read_text())
    if not isinstance(candidates, list):
        raise SystemExit("candidates file must be a JSON list of {name, rules, weights}")

    backend = create_backend("mlx")
    witek_index = None
    for c in candidates:
        if any(r.startswith("witek:") for r in c["rules"]):
            witek_index = WitekIndex.open()
            break
    print(f"Loaded {len(candidates)} candidate mixtures:")
    for c in candidates:
        print(f"  - {c['name']}: rules={c['rules']}  weights={c['weights']}")
    print()

    # Resolve rule LUTs once per candidate
    candidate_luts = []
    for c in candidates:
        luts = [fetch_lut_by_name(r, catalog=catalog, witek_index=witek_index) for r in c["rules"]]
        if abs(sum(c["weights"]) - 1.0) > 1e-6:
            raise SystemExit(f"weights for {c['name']} don't sum to 1: {sum(c['weights'])}")
        candidate_luts.append(luts)

    L = args.L
    max_steps = args.max_steps_mult * L

    # Aggregate per-trial outcomes per candidate.
    all_outcomes = {c["name"]: {"random": [], "adversarial": []} for c in candidates}
    median_steps = {c["name"]: {"random": [], "adversarial": []} for c in candidates}

    t0_total = time.time()
    for s_idx in range(args.n_seeds):
        seed = args.seed_base + s_idx
        rng_inits = np.random.default_rng(seed * 7919)

        # Build the SHARED inits for this seed.
        random_init, random_labels = _build_random_batch(
            L, args.delta, args.n_random_per_side, rng_inits,
        )
        adv_init, adv_labels, adv_names = build_extended_adversarial_inits(
            L, args.adv_densities, rng_inits,
        )

        print(f"=== seed {s_idx + 1}/{args.n_seeds} (seed_value={seed}) ===")
        print(f"  random batch: {len(random_labels)} trials,  adversarial: {len(adv_labels)} cases")

        for c, luts in zip(candidates, candidate_luts):
            t0 = time.time()
            r_res = evaluate_candidate(
                luts, c["weights"],
                init=random_init, labels=random_labels,
                L=L, max_steps=max_steps, backend=backend,
                seed=seed * 31 + s_idx,
            )
            a_res = evaluate_candidate(
                luts, c["weights"],
                init=adv_init, labels=adv_labels,
                L=L, max_steps=max_steps, backend=backend,
                seed=seed * 37 + s_idx,
            )
            elapsed = time.time() - t0
            all_outcomes[c["name"]]["random"].extend(r_res["correct_per_trial"])
            all_outcomes[c["name"]]["adversarial"].extend(a_res["correct_per_trial"])
            median_steps[c["name"]]["random"].append(r_res["median_steps"])
            median_steps[c["name"]]["adversarial"].append(a_res["median_steps"])
            print(f"  {c['name']:<22}  rand cc {r_res['n_correct']:>4}/{r_res['n_trials']}  "
                  f"adv cc {a_res['n_correct']:>3}/{a_res['n_trials']}  "
                  f"med_rand {r_res['median_steps']:.0f}  "
                  f"[{elapsed:.1f}s]")

    # Aggregate
    def binom_ci(k, n):
        if n == 0:
            return (float("nan"), float("nan"))
        p = k / n
        # Wilson 95% CI
        z = 1.96
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (center - half, center + half)

    summary = {}
    for cname in [c["name"] for c in candidates]:
        r = np.asarray(all_outcomes[cname]["random"], dtype=np.uint8)
        a = np.asarray(all_outcomes[cname]["adversarial"], dtype=np.uint8)
        rk = int(r.sum()); rn = int(r.size)
        ak = int(a.sum()); an = int(a.size)
        rci_lo, rci_hi = binom_ci(rk, rn)
        aci_lo, aci_hi = binom_ci(ak, an)
        summary[cname] = {
            "random_correct": rk, "random_total": rn,
            "random_cc": rk / rn if rn else float("nan"),
            "random_ci_95": [float(rci_lo), float(rci_hi)],
            "adversarial_correct": ak, "adversarial_total": an,
            "adversarial_cc": ak / an if an else float("nan"),
            "adversarial_ci_95": [float(aci_lo), float(aci_hi)],
            "median_steps_random_mean": float(np.mean(median_steps[cname]["random"])),
            "median_steps_adv_mean": float(np.mean(median_steps[cname]["adversarial"])),
        }

    # Paired McNemar tests
    paired = []
    names = [c["name"] for c in candidates]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for kind in ["random", "adversarial"]:
                xa = np.asarray(all_outcomes[a][kind], dtype=np.uint8)
                xb = np.asarray(all_outcomes[b][kind], dtype=np.uint8)
                # b01 = b right, a wrong; b10 = a right, b wrong
                b10 = int(((xa == 1) & (xb == 0)).sum())
                b01 = int(((xa == 0) & (xb == 1)).sum())
                # McNemar continuity-corrected (b10 + b01 small handled)
                n_disc = b10 + b01
                if n_disc == 0:
                    p = 1.0
                else:
                    # Exact binomial two-sided p-value: under H0, b10 ~ Binomial(n_disc, 0.5).
                    from math import comb
                    k = min(b10, b01)
                    p = 0.0
                    for j in range(0, k + 1):
                        p += comb(n_disc, j) * (0.5 ** n_disc)
                    p = min(2 * p, 1.0)
                paired.append({
                    "a": a, "b": b, "kind": kind,
                    "a_only_correct": b10,
                    "b_only_correct": b01,
                    "n_disagreements": n_disc,
                    "p_value_two_sided": p,
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "args": vars(args) | {"output": str(args.output),
                                "candidates_file": str(args.candidates_file)},
        "candidates": candidates,
        "summary": summary,
        "paired_mcnemar": paired,
        "elapsed_total_seconds": time.time() - t0_total,
    }, indent=2, default=str))

    # Print human-readable summary
    print()
    print("=== Per-candidate summary ===")
    print(f"{'name':<24} {'random_cc':>12} {'rand_ci':>22} {'adv_cc':>12} {'adv_ci':>22}")
    for cname in names:
        s = summary[cname]
        print(f"{cname:<24} "
              f"{s['random_cc']:>10.4f}  "
              f"[{s['random_ci_95'][0]:.4f}, {s['random_ci_95'][1]:.4f}] "
              f"{s['adversarial_cc']:>10.4f}  "
              f"[{s['adversarial_ci_95'][0]:.4f}, {s['adversarial_ci_95'][1]:.4f}]")

    print()
    print("=== Paired McNemar tests ===")
    print(f"{'comparison':<55} {'kind':<12} {'a-only':>7} {'b-only':>7} {'p-value':>10}")
    for r in paired:
        sig = " *" if r["p_value_two_sided"] < 0.05 else ""
        print(f"{r['a']} vs {r['b']:<{55 - len(r['a']) - 4}} "
              f"{r['kind']:<12} "
              f"{r['a_only_correct']:>7} {r['b_only_correct']:>7} "
              f"{r['p_value_two_sided']:>10.4f}{sig}")

    print(f"\nwrote {args.output}")
    print(f"total wall: {time.time() - t0_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
