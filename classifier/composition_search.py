"""Composition search for 2D density classification.

Matches the user's formulation:

    (F_1^{t_1} ∘ F_2^{t_2} ∘ ... ∘ F_k^{t_k})^T  then  local-majority^{T2}

where each F_i is a catalog NCCA. We restrict here to k = 2 (pairs) to keep
the search tractable, and sweep a few (t_1, t_2, T) budgets.

Each pair is evaluated by `min_correct_consensus_rate` across p = 0.49 and
p = 0.51 on multiple grid sizes. The aim is not to prove classification
perfect but to find any pair where both sides give nonzero correct consensus
under a pure finite-switch schedule.

This is the right target under the user's formulation. The effective
preprocessor (F_1^{t_1} F_2^{t_2})^T is a larger-radius NCCA not present
in the 133k-rule single-step catalog.
"""
from __future__ import annotations

import argparse
import itertools
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

from ca_search.binary_catalog import load_binary_catalog
from ca_search.density_classification import (
    moore_majority_rule_bits,
    von_neumann_majority_rule_bits,
)
from ca_search.simulator import create_backend


def evaluate_pair(
    F1: np.ndarray,
    F2: np.ndarray,
    G: np.ndarray,
    *,
    t1: int,
    t2: int,
    T: int,
    T2: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    backend,
) -> dict:
    batch, height, width = initial_states.shape
    L2 = height * width

    states = backend.asarray(initial_states, dtype="uint8")
    tiled_F1 = backend.asarray(np.tile(F1, (batch, 1)), dtype="uint8")
    tiled_F2 = backend.asarray(np.tile(F2, (batch, 1)), dtype="uint8")
    tiled_G = backend.asarray(np.tile(G, (batch, 1)), dtype="uint8")

    for _ in range(T):
        for _ in range(t1):
            states = backend.step_pairwise(states, tiled_F1)
        for _ in range(t2):
            states = backend.step_pairwise(states, tiled_F2)
    for _ in range(T2):
        states = backend.step_pairwise(states, tiled_G)

    final = backend.to_numpy(states)
    totals = final.reshape(batch, -1).sum(axis=1)
    all_zero = (totals == 0)
    all_one = (totals == L2)
    final_majority = (totals > (L2 / 2)).astype(np.uint8)

    correct = (all_zero & (labels == 0)) | (all_one & (labels == 1))
    consensus = all_zero | all_one
    return {
        "consensus_rate": float(consensus.mean()),
        "correct_consensus_rate": float(correct.mean()),
        "consensus_accuracy": (
            float(correct.sum()) / int(consensus.sum()) if int(consensus.sum()) > 0 else float("nan")
        ),
        "final_majority_accuracy": float((final_majority == labels).mean()),
    }


def build_initial_states(
    rng: np.random.Generator, grid: int, probabilities: list[float], trials_per_side: int
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rows = []
    labels_rows = []
    prob_col = []
    for p in probabilities:
        init = (rng.random((trials_per_side, grid, grid)) < p).astype(np.uint8)
        totals = init.reshape(trials_per_side, -1).sum(axis=1)
        labels = (totals > (grid * grid / 2)).astype(np.uint8)
        rows.append(init)
        labels_rows.append(labels)
        prob_col.extend([p] * trials_per_side)
    return np.concatenate(rows), np.concatenate(labels_rows), prob_col


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument(
        "--sids",
        nargs="+",
        required=True,
        help="Candidate sids to form ordered pairs from",
    )
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.49, 0.51])
    parser.add_argument("--trials-per-side", type=int, default=8)
    parser.add_argument(
        "--t-splits",
        nargs="+",
        type=str,
        default=["8x8x4", "4x4x8", "2x2x16"],
        help="Budgets t1xt2xT (F1^t1 then F2^t2, T outer repeats)",
    )
    parser.add_argument("--T2", type=int, default=128)
    parser.add_argument("--amplifier", choices=["moore", "vn"], default="vn")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("composition_search.json"))
    parser.add_argument("--top-report", type=int, default=30)
    parser.add_argument(
        "--include-self-pairs",
        action="store_true",
        help="Include pairs where F1 == F2 (same sid). Default: off.",
    )
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))

    # Resolve rule bits
    rule_bits: dict[str, np.ndarray] = {}
    for sid in args.sids:
        idx = catalog.resolve_rule_ref(sid)
        rule_bits[sid] = catalog.lut_bits[idx].astype(np.uint8)

    if args.amplifier == "moore":
        G = np.asarray(moore_majority_rule_bits(), dtype=np.uint8)
    else:
        G = np.asarray(von_neumann_majority_rule_bits(), dtype=np.uint8)

    # parse t-splits
    budgets: list[tuple[int, int, int]] = []
    for s in args.t_splits:
        parts = s.split("x")
        if len(parts) != 3:
            raise ValueError(f"Bad t-split: {s}")
        budgets.append((int(parts[0]), int(parts[1]), int(parts[2])))

    rng = np.random.default_rng(args.seed)
    initial_states, labels, prob_col = build_initial_states(
        rng, args.grid, args.probabilities, args.trials_per_side
    )
    backend = create_backend(args.backend)

    pair_list = list(itertools.product(args.sids, args.sids))
    if not args.include_self_pairs:
        pair_list = [pair for pair in pair_list if pair[0] != pair[1]]

    total_jobs = len(pair_list) * len(budgets)
    print(f"pairs={len(pair_list)} budgets={len(budgets)} total={total_jobs}")

    results: list[dict] = []
    t0 = time.time()
    done = 0
    for sid1, sid2 in pair_list:
        F1 = rule_bits[sid1]
        F2 = rule_bits[sid2]
        for (t1, t2, T) in budgets:
            # Evaluate per-p separately, then aggregate min across sides
            per_prob: list[dict] = []
            for p in args.probabilities:
                mask = np.asarray([float(pp) == float(p) for pp in prob_col], dtype=bool)
                init_p = initial_states[mask]
                labels_p = labels[mask]
                r = evaluate_pair(
                    F1, F2, G,
                    t1=t1, t2=t2, T=T, T2=args.T2,
                    initial_states=init_p,
                    labels=labels_p,
                    backend=backend,
                )
                r["probability"] = float(p)
                r["trials"] = int(mask.sum())
                per_prob.append(r)
            row = {
                "sid1": sid1,
                "sid2": sid2,
                "t1": t1, "t2": t2, "T": T, "T2": args.T2,
                "amplifier": args.amplifier,
                "grid": args.grid,
                "per_probability": per_prob,
                "min_correct_consensus_rate": min(x["correct_consensus_rate"] for x in per_prob),
                "min_consensus_rate": min(x["consensus_rate"] for x in per_prob),
                "min_final_majority_accuracy": min(x["final_majority_accuracy"] for x in per_prob),
            }
            results.append(row)
            done += 1
            if done % max(1, total_jobs // 20) == 0:
                print(f"  {done}/{total_jobs} ({time.time()-t0:.1f}s)")

    ranked = sorted(
        results,
        key=lambda r: (
            -r["min_correct_consensus_rate"],
            -r["min_final_majority_accuracy"],
            -r["min_consensus_rate"],
        ),
    )

    out = {
        "grid": args.grid,
        "probabilities": args.probabilities,
        "trials_per_side": args.trials_per_side,
        "T2": args.T2,
        "amplifier": args.amplifier,
        "candidate_sids": args.sids,
        "t_splits": args.t_splits,
        "ranked": ranked,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output}")

    print()
    print(f"=== Top {args.top_report} pairs by min correct consensus ===")
    for r in ranked[: args.top_report]:
        print(
            f"{r['sid1'][:16]:<16} + {r['sid2'][:16]:<16} "
            f"t1={r['t1']:2d} t2={r['t2']:2d} T={r['T']:2d} "
            f"  min_cc={r['min_correct_consensus_rate']:.3f}"
            f"  min_fma={r['min_final_majority_accuracy']:.3f}"
            f"  min_cons={r['min_consensus_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
