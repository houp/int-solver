"""Multi-seed validation of the top R_break candidates from
r_break_resume.

For each candidate (specified by global witek index), run the strict-CA
mixture (with R_break inserted as the 7th component at weight 0.05) on
random + adversarial inputs at several (L, delta) combinations across
multiple seeds.  Reports per-(L, delta) cc with confidence-interval-
style aggregation across seeds.
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
from witek_sampler import WitekIndex, _read_offsets_from_file, _decode_records_to_luts, RAW_RECORD_BYTES
from r_break_search import stage2_full_eval_one_rule
from strict_ca_classifier import resolve_rule_bits


def fetch_lut(global_index: int, witek_index: WitekIndex) -> np.ndarray:
    """Read a single rule by its global index from the witek/ corpus."""
    path, byte_offset = witek_index.locate(global_index)
    rec = _read_offsets_from_file(path, np.array([byte_offset], dtype=np.int64))
    luts = _decode_records_to_luts(rec)
    return luts[0]  # (512,) uint8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sid", default="sid:58ed6b657afb")
    parser.add_argument("--global-indices", type=int, nargs="+", required=True,
                        help="Global witek/ indices of the R_break candidates to validate")
    parser.add_argument("--Ls", type=int, nargs="+", default=[64, 96, 128, 192])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--n-random-per-side", type=int, default=32)
    parser.add_argument("--max-steps-mult", type=int, default=256)
    parser.add_argument("--output", type=Path,
                        default=Path("results/density_classification/2026-04-30/r_break_multi_seed_validate.json"))
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    catalog = load_binary_catalog(
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.bin"),
        str(_CATALOG_DIR / "expanded_property_panel_nonzero.json"),
    )
    F_bits = np.asarray(
        resolve_rule_bits(args.baseline_sid, catalog=catalog), dtype=np.uint8,
    )
    traffic_diag_luts = {
        name: resolve_rule_bits(name)
        for name in ["traffic_ne", "traffic_nw", "traffic_se", "traffic_sw"]
    }
    moore_lut = resolve_rule_bits("moore_maj")
    backend = create_backend("mlx")

    witek_index = WitekIndex.open()
    print(f"Witek corpus: {witek_index.n_total} rules across {len(witek_index.files)} files")

    all_results = []
    t0_total = time.time()
    for cand_i, gi in enumerate(args.global_indices):
        print(f"\n=== Candidate {cand_i + 1}/{len(args.global_indices)}: global index {gi} ===")
        lut = fetch_lut(gi, witek_index)
        per_config = []
        for L in args.Ls:
            for delta in args.deltas:
                for seed in args.seeds:
                    t0 = time.time()
                    res = stage2_full_eval_one_rule(
                        lut, F_bits=F_bits, traffic_diag_luts=traffic_diag_luts,
                        moore_lut=moore_lut, L=L, delta=delta,
                        n_random_per_side=args.n_random_per_side,
                        max_steps_mult=args.max_steps_mult,
                        seed=seed, backend=backend,
                    )
                    elapsed = time.time() - t0
                    row = {
                        "global_index": gi,
                        "L": L, "delta": delta, "seed": seed,
                        "random_cc": res["random_cc"],
                        "random_consensus": res["random_consensus"],
                        "random_total": res["random_total"],
                        "adversarial_cc": res["adversarial_cc"],
                        "adversarial_consensus": res["adversarial_consensus"],
                        "adversarial_total": res["adversarial_total"],
                        "random_median_steps": res["random_median_steps"],
                        "adversarial_median_steps": res["adversarial_median_steps"],
                        "elapsed_seconds": elapsed,
                    }
                    per_config.append(row)
                    print(f"  L={L:>3d} d={delta:.3f} seed={seed} "
                          f"adv={res['adversarial_cc']:.3f} ({res['adversarial_consensus']}/{res['adversarial_total']}) "
                          f"rand={res['random_cc']:.3f} ({res['random_consensus']}/{res['random_total']}) "
                          f"med_rand={res['random_median_steps']:.0f} "
                          f"[{elapsed:.1f}s]")
        all_results.extend(per_config)

    # Aggregate per (gi, L, delta) across seeds
    print(f"\n=== Aggregate (mean across seeds) ===")
    print(f"{'idx':>11} {'L':>4} {'delta':>6} {'rand_cc':>9} {'adv_cc':>9} {'med_rand':>9}")
    summary = {}
    for r in all_results:
        key = (r["global_index"], r["L"], r["delta"])
        summary.setdefault(key, []).append(r)
    aggregate_table = []
    for key, rows in sorted(summary.items()):
        gi, L, delta = key
        rand_mean = np.mean([rw["random_cc"] for rw in rows])
        adv_mean = np.mean([rw["adversarial_cc"] for rw in rows])
        rand_med_mean = np.mean([rw["random_median_steps"] for rw in rows
                                  if rw["random_median_steps"] != float("inf")])
        aggregate_table.append({
            "global_index": gi, "L": L, "delta": delta,
            "random_cc_mean": float(rand_mean),
            "adversarial_cc_mean": float(adv_mean),
            "random_median_steps_mean": float(rand_med_mean) if not np.isnan(rand_med_mean) else None,
            "n_seeds": len(rows),
        })
        print(f"{gi:>11} {L:>4} {delta:>6.3f} {rand_mean:>9.4f} {adv_mean:>9.4f} {rand_med_mean:>9.1f}")

    print(f"\nTotal elapsed: {time.time() - t0_total:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "args": vars(args) | {"output": str(args.output)},
        "per_config": all_results,
        "aggregate": aggregate_table,
    }, indent=2, default=str))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
