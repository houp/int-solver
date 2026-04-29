"""Faithful implementation of the Fates-Regnault (2016) 2D density classifier.

Reference: Fates & Regnault, "Solving the two-dimensional density
classification problem with two probabilistic cellular automata",
arXiv:1506.06653.

The construction is a two-phase schedule:

    Phase 1 (diffusion):    (F_X . R_184)^{T_1}
    Phase 2 (amplification): (G_X . R_232)^{T_2}

where:
- R_184 is the 1D traffic rule applied per row (rows independent),
- R_232 is the 1D majority-of-3 rule applied per row,
- F_X is a stochastic number-conserving "lane changing" rule that lets
  agents move vertically when blocked horizontally,
- G_X is a stochastic number-conserving "crowd avoidance" rule that lets
  isolated agents and 0/1 triplets diffuse vertically.

The paper reports 100% correct consensus at L = 50 with T_1 = T_2 = 2000
and at L = 100 with T_1 = T_2 = 8000 (each on 1000 random trials at
density rho = 0.49 / 0.51).

Coordinate convention:
- (i, j) = (row, col) with row increasing DOWNWARD.
- "above" = smaller row index = np.roll(s, 1, axis=Y_AXIS).
- The paper's "X(i, j-1)=1" condition (where j increases UPWARD in the
  paper) translates to "X[i_below, j_below] = 1" in our convention,
  i.e. the random bit at the SOURCE position of the moving agent.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# --- repo-root path bootstrap (keep BEFORE ca_search/local imports) ---
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
_CATALOG_DIR = _REPO_ROOT / "catalogs"
# ---------------------------------------------------------------------


# Y_AXIS / X_AXIS for clarity in np.roll:
#   states shape: (batch, H, W) where H = number of rows, W = number of cols.
#   Y_AXIS = 1 (rows). X_AXIS = 2 (cols).
Y_AXIS = 1
X_AXIS = 2


def step_R_184_2d(s: np.ndarray) -> np.ndarray:
    """1D rule 184 (traffic) applied per row, rows independent.

    Rule 184: x_i^{t+1} = x_{i-1}^t + x_i^t * x_{i+1}^t - x_{i-1}^t * x_i^t.
    Equivalently, in Boolean form:
        x_i^{t+1} = (x_{i-1} AND NOT x_i) OR (x_i AND x_{i+1}).

    Particle interpretation: an agent moves right iff the cell to its right
    is empty.
    """
    left = np.roll(s, 1, axis=X_AXIS)
    right = np.roll(s, -1, axis=X_AXIS)
    out = (left & (1 - s)) | (s & right)
    return out.astype(np.uint8)


def step_R_232_2d(s: np.ndarray) -> np.ndarray:
    """1D rule 232 (majority-of-3) applied per row, rows independent.

    Rule 232: x_i^{t+1} = majority(x_{i-1}, x_i, x_{i+1}).
    """
    left = np.roll(s, 1, axis=X_AXIS)
    right = np.roll(s, -1, axis=X_AXIS)
    total = left.astype(np.int16) + s.astype(np.int16) + right.astype(np.int16)
    return (total >= 2).astype(np.uint8)


def step_F_X(s: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Stochastic lane-changing rule.

    A cell at (i, j) acts as SOURCE of an upward move if:
      s[i, j] = 1, s[i-1, j] = 0 (north empty), s[i, j+1] = 1 (east occupied).
    A cell at (i, j) acts as DESTINATION of an upward move if:
      s[i, j] = 0, s[i+1, j] = 1 (south occupied),
      s[i+1, j+1] = 1 (south-east occupied).
    Both decisions are gated by the SAME random bit X[source_pos], which
    makes the move number-conserving per realization.
    """
    north = np.roll(s, 1, axis=Y_AXIS)
    south = np.roll(s, -1, axis=Y_AXIS)
    east = np.roll(s, -1, axis=X_AXIS)
    south_east = np.roll(np.roll(s, -1, axis=Y_AXIS), -1, axis=X_AXIS)

    # Source: at (i, j) with s=1, north=0, east=1.
    src_mask = (s == 1) & (north == 0) & (east == 1)
    # Destination: at (i, j) with s=0, south=1, south_east=1.
    dst_mask = (s == 0) & (south == 1) & (south_east == 1)

    # Random masks: source uses X[i, j] directly; destination uses X[i+1, j],
    # which is the random bit at the south (source) position.
    X_south = np.roll(X, -1, axis=Y_AXIS)

    out = s.copy()
    out[src_mask & (X == 1)] = 0
    out[dst_mask & (X_south == 1)] = 1
    return out.astype(np.uint8)


def step_G_X(s: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Stochastic crowd-avoidance rule.

    Two paired (source, destination) cases that move agents vertically:

    Case A ("isolated agent jumps up over an empty row"):
      Source at (i, j): s=1, N=0, NW=0, NE=0.
        With X[i, j]=1 the agent leaves: s -> 0.
      Destination at (i-1, j): s=0, W=0, E=0, S=1 (the agent below).
        With X[i, j]=1 (the source's X), the destination receives: s -> 1.

    Case B ("agent in a 1-1-1 horizontal triplet jumps up over empty above"):
      Source at (i, j): s=1, W=1, E=1, N=0.
        With X[i, j]=1 the agent leaves: s -> 0.
      Destination at (i-1, j): s=0, S=1, SW=1, SE=1.
        With X[i, j]=1 (source's X), s -> 1.

    Both cases are NC by construction (each X=1 event produces one 1->0
    and one 0->1 in paired positions).
    """
    north = np.roll(s, 1, axis=Y_AXIS)
    south = np.roll(s, -1, axis=Y_AXIS)
    west = np.roll(s, 1, axis=X_AXIS)
    east = np.roll(s, -1, axis=X_AXIS)
    nw = np.roll(np.roll(s, 1, axis=Y_AXIS), 1, axis=X_AXIS)
    ne = np.roll(np.roll(s, 1, axis=Y_AXIS), -1, axis=X_AXIS)
    sw = np.roll(np.roll(s, -1, axis=Y_AXIS), 1, axis=X_AXIS)
    se = np.roll(np.roll(s, -1, axis=Y_AXIS), -1, axis=X_AXIS)

    # Case A
    src_A = (s == 1) & (north == 0) & (nw == 0) & (ne == 0)
    # Destination at (i, j) for Case A: s=0, W=0, E=0, S=1, the source
    # is at (i+1, j), so the random bit is X_south.
    dst_A = (s == 0) & (west == 0) & (east == 0) & (south == 1)
    X_south = np.roll(X, -1, axis=Y_AXIS)

    # Case B: source at (i, j) with W=1, E=1, N=0.
    src_B = (s == 1) & (west == 1) & (east == 1) & (north == 0)
    # Destination at (i, j): s=0, S=1, SW=1, SE=1.
    dst_B = (s == 0) & (south == 1) & (sw == 1) & (se == 1)

    out = s.copy()
    out[(src_A | src_B) & (X == 1)] = 0
    out[(dst_A | dst_B) & (X_south == 1)] = 1
    return out.astype(np.uint8)


def run_fates_regnault(
    initial_states: np.ndarray,
    *,
    T1: int,
    T2: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Run the full Fates-Regnault schedule on a batch of inputs.

    initial_states: (batch, H, W) uint8 array.
    Returns: (final_states, diagnostics).
    """
    s = initial_states.astype(np.uint8).copy()
    initial_total = int(s.sum())

    # Phase 1: F_X then R_184, repeated T1 times
    for _ in range(T1):
        X = (rng.random(s.shape) < 0.5).astype(np.uint8)
        s = step_F_X(s, X)
        s = step_R_184_2d(s)

    # Phase 2: G_X then R_232, repeated T2 times
    for _ in range(T2):
        X = (rng.random(s.shape) < 0.5).astype(np.uint8)
        s = step_G_X(s, X)
        s = step_R_232_2d(s)

    final_total = int(s.sum())
    return s, {
        "T1": T1, "T2": T2,
        "initial_total": initial_total,
        "final_total": final_total,
        "density_drift": float(abs(final_total - initial_total) / max(1, initial_total)),
    }


def verify_number_conservation(L: int = 32, T_each: int = 100, n_trials: int = 8, seed: int = 0) -> bool:
    """Quick NC check: F_X and R_184 are NC; G_X and R_232 — R_232 is NOT NC.
    But the F_X-R_184 phase alone IS strictly NC. We verify Phase 1.
    """
    rng = np.random.default_rng(seed)
    init = (rng.random((n_trials, L, L)) < 0.5).astype(np.uint8)
    init_totals = init.reshape(n_trials, -1).sum(axis=1)

    s = init.copy()
    for _ in range(T_each):
        X = (rng.random(s.shape) < 0.5).astype(np.uint8)
        s = step_F_X(s, X)
        s = step_R_184_2d(s)
    after_phase1 = s.copy()
    after_totals = after_phase1.reshape(n_trials, -1).sum(axis=1)

    # Phase 1 must be strictly NC
    nc_ok = bool(np.all(after_totals == init_totals))
    return nc_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--probability", type=float, default=0.51)
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--T1", type=int, default=None,
                        help="Phase 1 length; default = 3 * L^2 (Fates-Regnault choice)")
    parser.add_argument("--T2", type=int, default=None,
                        help="Phase 2 length; default = 3 * L^2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-nc", action="store_true",
                        help="Run number-conservation sanity check first")
    parser.add_argument("--output", type=Path, default=Path("fates_regnault_demo.json"))
    args = parser.parse_args()

    if args.verify_nc:
        ok = verify_number_conservation(L=32, T_each=200, n_trials=8, seed=0)
        print(f"NC verification of (F_X . R_184)^200: {'OK' if ok else 'FAILED'}")
        if not ok:
            return 1

    L = args.L
    T1 = args.T1 if args.T1 is not None else 3 * L * L // 2
    T2 = args.T2 if args.T2 is not None else 3 * L * L // 2

    rng = np.random.default_rng(args.seed)
    init_above = (rng.random((args.n_trials, L, L)) < args.probability).astype(np.uint8)
    totals = init_above.reshape(args.n_trials, -1).sum(axis=1)
    label_above = (totals > L * L / 2).astype(np.uint8)

    rng_run = np.random.default_rng(args.seed + 1)
    t0 = time.time()
    final, diag = run_fates_regnault(init_above, T1=T1, T2=T2, rng=rng_run)
    elapsed = time.time() - t0

    final_totals = final.reshape(args.n_trials, -1).sum(axis=1)
    all_zero = (final_totals == 0)
    all_one = (final_totals == L * L)
    consensus = all_zero | all_one
    correct = (all_zero & (label_above == 0)) | (all_one & (label_above == 1))

    out = {
        "L": L,
        "T1": T1, "T2": T2,
        "probability": args.probability,
        "n_trials": args.n_trials,
        "consensus_rate": float(consensus.mean()),
        "correct_consensus_rate": float(correct.mean()),
        "elapsed_seconds": elapsed,
        "diag": diag,
    }
    print(f"L={L}, T1=T2={T1}, p={args.probability}, "
          f"trials={args.n_trials}: cc={out['correct_consensus_rate']:.3f} "
          f"({elapsed:.1f}s)")
    args.output.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
