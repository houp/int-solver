"""Analyze failure_diag_*.json + failure_snap_*.npz to characterize the
structural property of failing vs succeeding trials.

Prints per-failure features (initial density, after-F features, final
features, cycle period) and a compact ASCII render of each failing final
state.

Also renders a downscaled PNG of initial/after_F/final for each failing
trial, for visual inspection (via matplotlib if available).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np


def describe(rec: dict) -> str:
    d = rec["initial_density"]
    af = rec["after_F_features"]
    fn = rec["final_features"]
    return (
        f"p={rec['probability']:.2f} trial={rec['trial']:>3} "
        f"init_d={d:.4f} label={rec['initial_label']} "
        f"final_d={rec['final_density']:.4f} cycle={rec['cycle_period']}"
        f"\n  after_F: iface={af['iface_total']:>6d} "
        f"largest0={af['largest_zero_frac']:.3f} largest1={af['largest_one_frac']:.3f} "
        f"ncomp0={af['components_zero']:>4d} ncomp1={af['components_one']:>4d} "
        f"iface_h_frac={af['iface_fraction_horizontal']:.2f}"
        f"\n  final  : iface={fn['iface_total']:>6d} "
        f"largest0={fn['largest_zero_frac']:.3f} largest1={fn['largest_one_frac']:.3f} "
        f"ncomp0={fn['components_zero']:>4d} ncomp1={fn['components_one']:>4d} "
        f"iface_h_frac={fn['iface_fraction_horizontal']:.2f}"
    )


def ascii_render(state: np.ndarray, max_size: int = 64) -> str:
    """Downscale state to ≤ max_size x max_size using block-avg, render as chars."""
    h, w = state.shape
    step_h = max(1, h // max_size)
    step_w = max(1, w // max_size)
    downs = state[::step_h, ::step_w]
    lines = []
    for row in downs:
        lines.append("".join("#" if v else "." for v in row))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", type=Path, required=True)
    parser.add_argument("--snaps", type=Path, required=True)
    parser.add_argument("--show-ascii", action="store_true")
    parser.add_argument("--ascii-max-size", type=int, default=48)
    args = parser.parse_args()

    diag = json.loads(args.diag.read_text())
    snaps = np.load(args.snaps)
    initial = snaps["initial"]
    after_F = snaps["after_F"]
    final = snaps["final"]
    labels = snaps["labels"]

    failures = [r for r in diag["per_trial"] if not r["correct"]]
    successes = [r for r in diag["per_trial"] if r["correct"]]

    print(f"=== failure_analysis ===")
    print(f"sid={diag['sid'][:16]}  grid={diag['grid']}  T1={diag['T1']}  T2={diag['T2']}  "
          f"amp={diag['amplifier']}")
    print(f"total={diag['total_trials']} failures={len(failures)} successes={len(successes)}")
    print()

    # Aggregate: how do feature distributions differ between fail and succeed?
    def collect(records, key_path):
        keys = key_path.split(".")
        vals = []
        for r in records:
            cur = r
            for k in keys:
                cur = cur[k]
            vals.append(float(cur))
        return vals

    def stat(records, key_path):
        vals = collect(records, key_path)
        if not vals:
            return float("nan"), float("nan"), float("nan")
        return (
            statistics.fmean(vals),
            statistics.median(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0.0,
        )

    feats = [
        "initial_density",
        "after_F_features.iface_total",
        "after_F_features.largest_zero_frac",
        "after_F_features.largest_one_frac",
        "after_F_features.components_zero",
        "after_F_features.components_one",
        "after_F_features.iface_fraction_horizontal",
        "final_features.iface_total",
        "final_features.largest_zero_frac",
        "final_features.largest_one_frac",
        "final_features.components_zero",
        "final_features.components_one",
    ]

    print(f"{'feature':<50s} {'fail_mean':>10s} {'succ_mean':>10s} {'fail_med':>10s} {'succ_med':>10s}")
    print("-" * 95)
    for f in feats:
        fm, fmed, _ = stat(failures, f)
        sm, smed, _ = stat(successes, f)
        print(f"{f:<50s} {fm:>10.3f} {sm:>10.3f} {fmed:>10.3f} {smed:>10.3f}")

    print()
    print(f"=== Each failing trial ===")
    for r in failures:
        print(describe(r))
        if args.show_ascii:
            # Find trial index in the flat snapshot bundle
            # The bundle order is: probabilities × trials_per_side
            tpsc = diag["trials_per_side"]
            prob_idx = diag["probabilities"].index(r["probability"])
            flat_idx = prob_idx * tpsc + r["trial"]
            print("INITIAL:")
            print(ascii_render(initial[flat_idx], args.ascii_max_size))
            print("AFTER F:")
            print(ascii_render(after_F[flat_idx], args.ascii_max_size))
            print("FINAL:")
            print(ascii_render(final[flat_idx], args.ascii_max_size))
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
