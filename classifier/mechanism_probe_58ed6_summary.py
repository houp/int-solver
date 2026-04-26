"""Summarize the P1.1 mechanism-probe output.

Reads mechanism_58ed6.json and prints a compact time-series table per
(grid, probability) run, averaging metrics across trials.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def fmt(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def summarize_run(run: dict) -> None:
    grid = run["grid"]
    prob = run["probability"]
    print(f"\n=== grid={grid}x{grid}  p={prob}  trials={run['trials']}  seed={run['seed']} ===")
    # header
    cols = [
        ("step", 5),
        ("density", 9),
        ("moore_agree", 11),
        ("vn_agree", 8),
        ("decode_strict", 13),
        ("local_var", 9),
        ("iface/L^2", 9),
        ("comp_cnt", 8),
        ("largest_frac", 12),
        ("spec_lo/hi", 11),
    ]
    header = " ".join(name.ljust(w) for name, w in cols)
    print(header)
    print("-" * len(header))

    L = grid
    L2 = L * L

    for step_str, metrics in run["metrics_by_step"].items():
        step = int(step_str)
        density = statistics.fmean(metrics["densities"])
        moore = statistics.fmean(metrics["moore_majority_agreement"])
        vn = statistics.fmean(metrics["vn_majority_agreement"])
        decode = statistics.fmean(metrics["decodability_strict"])
        lvar = statistics.fmean(metrics["local_density_variance"])
        iface = statistics.fmean(metrics["interface_length"]) / L2
        ncomp = statistics.fmean(metrics["num_majority_components"])
        largest = statistics.fmean(metrics["largest_majority_component_fraction"])
        # ratio of low-k power to high-k power across radial bins
        spectra = metrics["radial_spectrum"]
        lo_band = statistics.fmean(
            [statistics.fmean(s[:4]) for s in spectra]
        )
        hi_band = statistics.fmean(
            [statistics.fmean(s[-4:]) for s in spectra]
        )
        ratio = lo_band / hi_band if hi_band > 0 else float("nan")
        row = [
            str(step).ljust(cols[0][1]),
            fmt(density).ljust(cols[1][1]),
            fmt(moore).ljust(cols[2][1]),
            fmt(vn).ljust(cols[3][1]),
            fmt(decode).ljust(cols[4][1]),
            fmt(lvar, 5).ljust(cols[5][1]),
            fmt(iface, 4).ljust(cols[6][1]),
            fmt(ncomp, 1).ljust(cols[7][1]),
            fmt(largest, 4).ljust(cols[8][1]),
            fmt(ratio, 2).ljust(cols[9][1]),
        ]
        print(" ".join(row))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("mechanism_58ed6.json"))
    args = parser.parse_args()
    data = json.loads(args.path.read_text())
    print(f"rule: {data['rule_sid']} (legacy_index={data['rule_legacy_index']})")
    for run in data["runs"]:
        summarize_run(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
