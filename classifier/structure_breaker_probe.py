"""Test whether adding 'structure-breaker' rules to the rebalanced
mixture improves performance on the parameterized adversarial battery,
particularly on the failure-mode families (half_h, voronoi, fourier).

Hypothesis: the recommended 6-rule mixture (F_0 + 4 diagonal traffic +
M_Moore9) fails on inputs with strong large-scale spatial structure
because the local Moore-9 mean-field assumption breaks down. Adding
NCCAs that transport mass across the lattice in different directions
might help.

Rule sets evaluated (each weighted with the SAME structure as the
rebalanced 6-rule mixture, with any new rules sharing equally a
fraction taken from p_M):

  set_A: 6-rule rebalanced (control).
  set_B: + 4 cardinal traffic (N/S/E/W). p_M -> 0.5*p_M; new rules
         each get 0.5*p_M / 4.
  set_C: + 4 shift rules (N/S/E/W rigid translations, NOT NCCAs but
         density-preserving). Same weight scheme as set_B.
  set_D: + both 4 cardinal traffic + 4 shifts. p_M -> 0.5*p_M; new
         rules each get 0.5*p_M / 8.
  set_E: drop F_0 entirely, redistribute its weight uniformly among
         the four diagonal traffic rules. (Sanity check: does F_0
         actually help on adversarial?)

For each rule set we evaluate on the same parameterized adversarial
battery (444 cases x 4 seeds = 1776 trials) and report the
label-symmetric correctness per family.
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
from parameterized_adversarial import build_parameterized_battery


# Rebalanced 6-rule mixture
W_F0     = 0.041
W_NE     = 0.049
W_NW     = 0.126
W_SE     = 0.017
W_SW     = 0.579
W_MAJ    = 0.188


def make_rule_set(name: str):
    """Return (rule_names, weights) tuple."""
    if name == "A_rebalanced":
        rules = ["sid:58ed6b657afb",
                 "traffic_ne", "traffic_nw", "traffic_se", "traffic_sw",
                 "moore_maj"]
        w = [W_F0, W_NE, W_NW, W_SE, W_SW, W_MAJ]
    elif name == "B_plus_card_traffic":
        rules = ["sid:58ed6b657afb",
                 "traffic_ne", "traffic_nw", "traffic_se", "traffic_sw",
                 "traffic_n", "traffic_s", "traffic_e", "traffic_w",
                 "moore_maj"]
        ext = 0.5 * W_MAJ / 4
        w = [W_F0, W_NE, W_NW, W_SE, W_SW, ext, ext, ext, ext, 0.5 * W_MAJ]
    elif name == "C_plus_shifts":
        rules = ["sid:58ed6b657afb",
                 "traffic_ne", "traffic_nw", "traffic_se", "traffic_sw",
                 "shift_n", "shift_s", "shift_e", "shift_w",
                 "moore_maj"]
        ext = 0.5 * W_MAJ / 4
        w = [W_F0, W_NE, W_NW, W_SE, W_SW, ext, ext, ext, ext, 0.5 * W_MAJ]
    elif name == "D_plus_card_traffic_and_shifts":
        rules = ["sid:58ed6b657afb",
                 "traffic_ne", "traffic_nw", "traffic_se", "traffic_sw",
                 "traffic_n", "traffic_s", "traffic_e", "traffic_w",
                 "shift_n", "shift_s", "shift_e", "shift_w",
                 "moore_maj"]
        ext = 0.5 * W_MAJ / 8
        w = [W_F0, W_NE, W_NW, W_SE, W_SW] + [ext] * 8 + [0.5 * W_MAJ]
    elif name == "E_drop_F0":
        # Redistribute W_F0 to the four diagonal traffic rules
        extra = W_F0 / 4
        rules = ["traffic_ne", "traffic_nw", "traffic_se", "traffic_sw",
                 "moore_maj"]
        w = [W_NE + extra, W_NW + extra, W_SE + extra, W_SW + extra, W_MAJ]
    elif name == "F_no_diag_traffic":
        # Cardinal traffic only, drop the diagonals
        diag_total = W_NE + W_NW + W_SE + W_SW
        rules = ["sid:58ed6b657afb",
                 "traffic_n", "traffic_s", "traffic_e", "traffic_w",
                 "moore_maj"]
        w = [W_F0, diag_total / 4, diag_total / 4, diag_total / 4, diag_total / 4, W_MAJ]
    else:
        raise ValueError(f"unknown rule set: {name}")
    s = sum(w)
    w = [x / s for x in w]
    return rules, w


def label_symmetric_cc(correct, labels):
    correct = np.asarray(correct, dtype=bool)
    labels = np.asarray(labels)
    m0 = labels == 0; m1 = labels == 1
    cc0 = float(correct[m0].mean()) if m0.any() else 1.0
    cc1 = float(correct[m1].mean()) if m1.any() else 1.0
    return cc0, cc1, min(cc0, cc1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-sets", nargs="+",
                        default=["A_rebalanced", "B_plus_card_traffic",
                                 "C_plus_shifts", "D_plus_card_traffic_and_shifts",
                                 "E_drop_F0", "F_no_diag_traffic"])
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.51, 0.52, 0.55])
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--max-steps-mult", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=2026)
    parser.add_argument("--output", type=Path,
                        default=Path("results/density_classification/2026-05-01/structure_breaker_probe.json"))
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    catalog = load_binary_catalog(
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.bin"),
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.json"),
    )
    backend = create_backend("mlx")
    L = args.L; max_steps = args.max_steps_mult * L

    # Resolve rule LUTs once
    rule_set_resolved = {}
    for s in args.rule_sets:
        rules, w = make_rule_set(s)
        luts = [resolve_rule_bits(r, catalog=catalog) for r in rules]
        rule_set_resolved[s] = (rules, w, luts)
        print(f"  set {s}: {len(rules)} rules, w sum = {sum(w):.4f}")

    # Per (set, family) accumulator
    per_set_records = defaultdict(list)

    for s_idx in range(args.n_seeds):
        seed = args.seed_base + s_idx
        rng_inits = np.random.default_rng(seed * 7919)
        battery, labels, names = build_parameterized_battery(
            L, args.densities, rng_inits)
        print(f"\n=== seed {s_idx + 1}/{args.n_seeds} (seed={seed}, n_cases={len(names)}) ===")

        for s_name in args.rule_sets:
            rules, w, luts = rule_set_resolved[s_name]
            t0 = time.time()
            rng_run = np.random.default_rng(seed * 31 + s_idx + hash(s_name) % 100)
            out = run_strict_ca(
                battery, rule_lut_list=luts, probabilities=w,
                max_steps=max_steps, until_consensus=True,
                backend=backend, rng=rng_run,
            )
            correct = _correct_consensus(out["final_totals"], labels, L * L)
            cc0, cc1, mn = label_symmetric_cc(correct, labels)
            elapsed = time.time() - t0
            print(f"  {s_name:<32}  cc0={cc0:.4f} cc1={cc1:.4f} min={mn:.4f}  "
                  f"failures={(~correct.astype(bool)).sum()}  [{elapsed:.1f}s]")
            for k, name in enumerate(names):
                per_set_records[s_name].append({
                    "seed_idx": s_idx, "seed_value": seed,
                    "case_name": name, "label": int(labels[k]),
                    "correct": bool(correct[k]),
                })

    # Aggregate per set: label-symmetric and per-family
    def family_of(name):
        base = name.rsplit("_d", 1)[0]
        for sep in ["_p", "_b", "_w", "_n", "_kx", "_e"]:
            base = base.split(sep)[0]
        return base.rstrip("_")

    print("\n=== Aggregate per rule set ===")
    aggregate = {}
    for s_name in args.rule_sets:
        recs = per_set_records[s_name]
        labels_a = np.array([r["label"] for r in recs])
        correct_a = np.array([r["correct"] for r in recs])
        cc0, cc1, mn = label_symmetric_cc(correct_a, labels_a)
        # Per-family
        fam = defaultdict(lambda: [0, 0])
        for r in recs:
            f = family_of(r["case_name"])
            fam[f][0] += int(r["correct"]); fam[f][1] += 1
        fam_cc = {k: v[0] / v[1] for k, v in fam.items()}
        aggregate[s_name] = {"cc0": cc0, "cc1": cc1, "min": mn,
                             "n_correct": int(correct_a.sum()),
                             "n_total": len(correct_a),
                             "fam_cc": fam_cc}
        print(f"\n  {s_name}: cc0={cc0:.4f}  cc1={cc1:.4f}  min={mn:.4f}  "
              f"({int(correct_a.sum())}/{len(correct_a)})")
        for f, c in sorted(fam_cc.items()):
            star = " <" if c < 0.5 else ""
            print(f"    {f:<22}  {c:.4f}{star}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "args": vars(args) | {"output": str(args.output)},
        "rule_sets": {s: {"rules": r[0], "weights": r[1]}
                      for s, r in rule_set_resolved.items()},
        "aggregate": aggregate,
        "all_records": dict(per_set_records),
    }, indent=2, default=str))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
