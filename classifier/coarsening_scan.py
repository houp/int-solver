"""Scan a catalog of NCCAs for coarsening / canonicalizing behavior.

For each candidate rule, run it alone on a random near-critical lattice for
a fixed number of steps, then measure:

- asymptotic_decodability: fraction of sites where local Moore majority equals
  the true global majority label (averaged across trials on both sides of
  p = 1/2)
- final_interface_fraction: (horizontal + vertical unequal-neighbor pairs) / L^2
  (low means phase-separated; high means fragmented)
- largest_component_fraction: largest connected cluster of the global-majority
  value, as a fraction of lattice area

Rules are ranked by asymptotic_decodability (primary) and 1 - interface_fraction
(secondary). A Fukś-style candidate needs BOTH to be near 1.

This scan exploits the batched pairwise simulator to run many rules in
parallel on separate initial configurations.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


# --- repo-root path bootstrap (keep BEFORE ca_search/local imports) ---
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
_CATALOG_DIR = _REPO_ROOT / "catalogs"
# ---------------------------------------------------------------------

import numpy as np
from scipy import ndimage

from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend


def moore_majority_batch(states: np.ndarray) -> np.ndarray:
    x = np.roll(np.roll(states, 1, axis=1), 1, axis=2)
    y = np.roll(states, 1, axis=1)
    z = np.roll(np.roll(states, 1, axis=1), -1, axis=2)
    t = np.roll(states, 1, axis=2)
    u = states
    w = np.roll(states, -1, axis=2)
    a = np.roll(np.roll(states, -1, axis=1), 1, axis=2)
    b = np.roll(states, -1, axis=1)
    c = np.roll(np.roll(states, -1, axis=1), -1, axis=2)
    s = (x + y + z + t + u + w + a + b + c).astype(np.int16)
    return (s >= 5).astype(np.uint8)


def interface_fraction_batch(states: np.ndarray) -> np.ndarray:
    """Per-trial (iface_h + iface_v) / L^2."""
    h = (states != np.roll(states, -1, axis=1)).sum(axis=(1, 2))
    v = (states != np.roll(states, -1, axis=2)).sum(axis=(1, 2))
    L2 = states.shape[1] * states.shape[2]
    return (h + v) / float(L2)


def largest_majority_component(state: np.ndarray, value: int) -> float:
    mask = (state == value).astype(np.uint8)
    labeled, n = ndimage.label(mask)
    if n == 0:
        return 0.0
    sizes = np.bincount(labeled.ravel())
    return float(sizes[1:].max()) / float(state.size)


def scan_rules(
    catalog,
    indices: np.ndarray,
    *,
    grid: int,
    steps: int,
    trials_per_side: int,
    seed: int,
    backend_name: str,
    chunk_size: int = 512,
) -> list[dict]:
    """Scan each rule on 2*trials_per_side trials (half at p=0.49, half at p=0.51).

    Each rule gets its own, independently sampled set of initial states.
    """
    backend = create_backend(backend_name)
    rng = np.random.default_rng(seed)
    n_rules = len(indices)
    total_trials = 2 * trials_per_side

    results: list[dict] = []
    t0 = time.time()
    for chunk_start in range(0, n_rules, chunk_size):
        chunk = indices[chunk_start : chunk_start + chunk_size]
        nc = len(chunk)

        # Build initial states: one per rule-trial pair, shape (nc*total_trials, grid, grid)
        batch = nc * total_trials
        # For each rule, half trials at p=0.49 and half at p=0.51
        probs = np.tile(
            np.concatenate(
                [
                    np.full(trials_per_side, 0.49),
                    np.full(trials_per_side, 0.51),
                ]
            ),
            nc,
        )
        init = (rng.random((batch, grid, grid)) < probs[:, None, None]).astype(np.uint8)

        # Global labels (using initial density)
        totals = init.reshape(batch, -1).sum(axis=1)
        labels = (totals > (grid * grid / 2)).astype(np.uint8)

        # Tile rule bits to the batch: each block of total_trials rows shares one rule
        rule_bits = catalog.lut_bits[chunk].astype(np.uint8)  # (nc, 512)
        tiled_rules = np.repeat(rule_bits, total_trials, axis=0)  # (batch, 512)

        states = backend.asarray(init, dtype="uint8")
        tiled = backend.asarray(tiled_rules, dtype="uint8")

        for _ in range(steps):
            states = backend.step_pairwise(states, tiled)

        final = backend.to_numpy(states)

        # Compute decodability per trial
        moore = moore_majority_batch(final)
        decode = (moore == labels[:, None, None]).mean(axis=(1, 2))
        iface = interface_fraction_batch(final)

        # Aggregate per rule
        decode_per_rule = decode.reshape(nc, total_trials).mean(axis=1)
        decode_min_side = decode.reshape(nc, 2, trials_per_side).mean(axis=2).min(axis=1)
        iface_per_rule = iface.reshape(nc, total_trials).mean(axis=1)

        # Per-trial largest-component is expensive; sample one trial per rule for it
        largest_per_rule = np.zeros(nc, dtype=np.float32)
        for i in range(nc):
            # pick the first p=0.49 trial for this rule
            trial_idx = i * total_trials
            largest_per_rule[i] = largest_majority_component(
                final[trial_idx], int(labels[trial_idx])
            )

        for i, ridx in enumerate(chunk):
            results.append(
                {
                    "stable_index": int(catalog.stable_indices[ridx]),
                    "stable_id": str(catalog.stable_ids[ridx][:16]),
                    "legacy_index": int(ridx),
                    "decodability_mean": float(decode_per_rule[i]),
                    "decodability_min_side": float(decode_min_side[i]),
                    "interface_fraction": float(iface_per_rule[i]),
                    "largest_component_fraction_sample": float(largest_per_rule[i]),
                }
            )

        elapsed = time.time() - t0
        done = chunk_start + nc
        print(f"  {done}/{n_rules} rules scanned ({elapsed:.1f}s elapsed)")

    return results


def select_indices_by_property(
    catalog, property_names: list[str]
) -> np.ndarray:
    mask_bits = 0
    for name in property_names:
        match = [p for p in catalog.properties if p.name == name]
        if not match:
            raise KeyError(f"property not found: {name}")
        mask_bits |= 1 << match[0].bit
    sel = np.nonzero((catalog.masks & mask_bits) == mask_bits)[0]
    return sel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--properties", nargs="+", default=None,
                        help="Require these property tags ANDed (e.g. outer_monotone orthogonal_monotone diagonal_monotone)")
    parser.add_argument("--random-sample", type=int, default=0,
                        help="If > 0, uniformly sample this many rules from the (filtered) set")
    parser.add_argument("--sample-seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to the first N indices after filtering/sampling (0 = no limit)")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--trials-per-side", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("coarsening_scan.json"))
    parser.add_argument("--top-report", type=int, default=30)
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    print(f"catalog: {len(catalog.ids)} rules, {len(catalog.properties)} properties")

    if args.properties:
        indices = select_indices_by_property(catalog, args.properties)
        print(f"filtered by {args.properties}: {len(indices)} rules")
    else:
        indices = np.arange(len(catalog.ids))

    if args.random_sample > 0 and len(indices) > args.random_sample:
        rng = np.random.default_rng(args.sample_seed)
        indices = rng.choice(indices, size=args.random_sample, replace=False)
        indices.sort()
        print(f"random-sampled to {len(indices)} rules (seed={args.sample_seed})")

    if args.limit > 0 and len(indices) > args.limit:
        indices = indices[: args.limit]
        print(f"limited to {len(indices)} rules")

    results = scan_rules(
        catalog,
        indices,
        grid=args.grid,
        steps=args.steps,
        trials_per_side=args.trials_per_side,
        seed=args.seed,
        backend_name=args.backend,
        chunk_size=args.chunk_size,
    )

    # Rank: (decodability_min_side desc), then (interface_fraction asc)
    ranked = sorted(
        results,
        key=lambda r: (-r["decodability_min_side"], r["interface_fraction"]),
    )

    out = {
        "grid": args.grid,
        "steps": args.steps,
        "trials_per_side": args.trials_per_side,
        "seed": args.seed,
        "properties_filter": args.properties or [],
        "random_sample": args.random_sample,
        "limit": args.limit,
        "total_rules_scanned": len(results),
        "ranked": ranked,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output}")

    print()
    print(f"=== Top {args.top_report} by min-side decodability ===")
    header = "sid               dec_min  dec_mean  iface_frac  largest_frac"
    print(header)
    print("-" * len(header))
    for r in ranked[: args.top_report]:
        print(
            f"{r['stable_id']:<18}"
            f"{r['decodability_min_side']:.4f}  "
            f"{r['decodability_mean']:.4f}  "
            f"{r['interface_fraction']:.4f}      "
            f"{r['largest_component_fraction_sample']:.4f}"
        )

    print()
    print(f"=== Lowest interface fraction (potential coarseners) ===")
    by_iface = sorted(results, key=lambda r: r["interface_fraction"])
    for r in by_iface[:20]:
        print(
            f"{r['stable_id']:<18}"
            f"iface={r['interface_fraction']:.4f}  "
            f"largest={r['largest_component_fraction_sample']:.4f}  "
            f"dec_min={r['decodability_min_side']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
