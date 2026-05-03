"""Active adversarial search using the parameterized battery.

For each candidate mixture, evaluate the FULL parameterized adversarial
battery (444+ cases per density set) at L=64 across multiple seeds.
Reports per-family breakdown and identifies any case where the
recommended mixture FAILS.
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
from parameterized_adversarial import build_parameterized_battery


def fetch_lut_by_name(name, *, catalog, witek_index=None):
    if name.startswith("witek:"):
        gi = int(name.split(":", 1)[1])
        if witek_index is None:
            raise RuntimeError("witek index needed but not loaded")
        path, byte_offset = witek_index.locate(gi)
        rec = _read_offsets_from_file(path, np.array([byte_offset], dtype=np.int64))
        return list(_decode_records_to_luts(rec)[0])
    return resolve_rule_bits(name, catalog=catalog)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-file", type=Path, required=True)
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.51, 0.52, 0.55])
    parser.add_argument("--n-seeds", type=int, default=4)
    parser.add_argument("--max-steps-mult", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    catalog = load_binary_catalog(
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.bin"),
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.json"),
    )
    candidates = json.loads(args.candidates_file.read_text())
    backend = create_backend("mlx")
    witek_index = None
    for c in candidates:
        if any(r.startswith("witek:") for r in c["rules"]):
            witek_index = WitekIndex.open(); break

    candidate_luts = [
        [fetch_lut_by_name(r, catalog=catalog, witek_index=witek_index) for r in c["rules"]]
        for c in candidates
    ]

    L = args.L
    max_steps = args.max_steps_mult * L

    # Build the battery once per seed
    all_records = []
    t0_total = time.time()

    for s_idx in range(args.n_seeds):
        seed = args.seed_base + s_idx
        rng_inits = np.random.default_rng(seed * 7919)
        battery, labels, names = build_parameterized_battery(
            L, args.densities, rng_inits,
        )
        print(f"=== seed {s_idx + 1}/{args.n_seeds} (seed={seed}, n_cases={len(names)}) ===")

        for c, luts in zip(candidates, candidate_luts):
            t0 = time.time()
            rng_run = np.random.default_rng(seed * 31 + s_idx + hash(c["name"]) % 100)
            out = run_strict_ca(
                battery, rule_lut_list=luts, probabilities=list(c["weights"]),
                max_steps=max_steps, until_consensus=True,
                backend=backend, rng=rng_run,
            )
            correct = _correct_consensus(out["final_totals"], labels, L * L)
            elapsed = time.time() - t0
            n_correct = int(correct.sum())
            n_total = len(labels)
            failed_names = [names[i] for i, c_ in enumerate(correct) if not c_]
            print(f"  {c['name']:<28}  cc={n_correct}/{n_total} ({n_correct / n_total:.4f})  "
                  f"failures={len(failed_names)}  [{elapsed:.1f}s]")
            for r_idx, name in enumerate(names):
                all_records.append({
                    "seed_idx": s_idx,
                    "seed_value": seed,
                    "candidate": c["name"],
                    "case_name": name,
                    "label": int(labels[r_idx]),
                    "correct": bool(correct[r_idx]),
                })

    # Aggregate
    print()
    print("=== Aggregate per candidate (across seeds, full battery) ===")
    per_cand = {}
    for r in all_records:
        per_cand.setdefault(r["candidate"], []).append(r["correct"])
    for c, lst in per_cand.items():
        cc = sum(lst) / len(lst)
        n_correct = sum(lst); n_total = len(lst)
        print(f"  {c:<28}  cc={n_correct}/{n_total} ({cc:.4f})")

    # Per-family breakdown
    def family_of(name):
        # strip trailing _d... and parameter suffix
        base = name.rsplit("_d", 1)[0]
        for sep in ["_p", "_b", "_w", "_n", "_kx", "_e"]:
            base = base.split(sep)[0]
        return base.rstrip("_")

    print()
    print("=== Per-family failure count for the rebalanced mixture ===")
    fam_records = {}
    target_cand = None
    for c in candidates:
        if "rebalanced" in c["name"]:
            target_cand = c["name"]; break
    if target_cand is None:
        target_cand = candidates[-1]["name"]
    for r in all_records:
        if r["candidate"] != target_cand:
            continue
        f = family_of(r["case_name"])
        fam_records.setdefault(f, [0, 0])
        fam_records[f][0] += int(r["correct"])
        fam_records[f][1] += 1
    for f, (k, n) in sorted(fam_records.items()):
        if k < n:
            star = " <-- FAILURES"
        else:
            star = ""
        print(f"  {f:<22}  {k}/{n} ({k/n:.4f}){star}")

    # Identify failing inputs (any candidate, any seed)
    failing = [r for r in all_records if not r["correct"]]
    print()
    print(f"Total failures across all candidates and seeds: {len(failing)}")
    if failing:
        # Show breakdown by candidate
        by_cand = {}
        for r in failing:
            by_cand.setdefault(r["candidate"], []).append(r)
        for c, lst in by_cand.items():
            print(f"  {c}: {len(lst)} failures across {len(lst)} unique cases")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "args": vars(args) | {"output": str(args.output),
                                "candidates_file": str(args.candidates_file)},
        "candidates": candidates,
        "all_records": all_records,
        "per_candidate_aggregate": {
            c: {"correct": sum(lst), "total": len(lst), "cc": sum(lst) / len(lst)}
            for c, lst in per_cand.items()
        },
        "elapsed_total_seconds": time.time() - t0_total,
    }, indent=2, default=str))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
