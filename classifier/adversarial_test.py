"""Test the current best schedule on adversarial initial configurations.

These are non-random starting states designed to probe specific failure modes:

- striped_horizontal / striped_vertical: thin horizontal or vertical stripes
  at near-critical density. Majority's stripe fixed points.
- checkerboard: exact checkerboard with specific imbalance.
- block_checker: 2x2 blocks arranged in a checkerboard.
- two_blobs: one large rectangular minority region + random background.
- one_big_blob: circular minority region of specified radius.
- quadrant_imbalance: one quadrant of the torus flipped to majority color.
- near_critical_random: random with density exactly (1/L^2) away from 1/2.

For each initial state, we run the schedule and check whether it reaches
all-correct consensus. A "universal" classifier should handle all of these.
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

from amplifier_library import apply_radius_step_mlx
from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend


def make_adversarial_batch(L: int, target_label: int, rng: np.random.Generator) -> dict:
    """Return dict name -> (initial_state, label). Label is the global majority."""
    out = {}

    # --- stripes ---
    # Horizontal stripes of width 1: alternate rows 0 and 1. If L even, density 0.5 exactly.
    # We use width-2 stripes to get density 0.5 but avoid being checkerboard-like.
    # To ensure non-degenerate majority, adjust ONE cell to tip the balance.
    base = np.zeros((L, L), dtype=np.uint8)
    base[::2] = 1
    # density = 0.5. Flip one cell to make majority match target_label.
    if target_label == 1:
        base[1, 0] = 1  # flip one 0 to 1 -> majority 1
    else:
        base[0, 0] = 0  # flip one 1 to 0 -> majority 0
    out["stripes_h_w1"] = base.copy()

    # Vertical stripes width 1
    base = np.zeros((L, L), dtype=np.uint8)
    base[:, ::2] = 1
    if target_label == 1:
        base[0, 1] = 1
    else:
        base[0, 0] = 0
    out["stripes_v_w1"] = base.copy()

    # --- checkerboard ---
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    cb = ((ii + jj) % 2).astype(np.uint8)
    # cb has density 0.5 exactly. Flip one cell.
    if target_label == 1:
        cb[0, 1] = 1
    else:
        cb[0, 0] = 0
    out["checkerboard"] = cb.copy()

    # --- block checker 2x2 ---
    bc = ((ii // 2 + jj // 2) % 2).astype(np.uint8)
    if target_label == 1:
        bc[0, 2] = 1
    else:
        bc[0, 0] = 0
    out["block_checker_2x2"] = bc.copy()

    # --- one big blob (circle of minority in sea of majority) ---
    # radius chosen so minority density ~ 0.45-0.49 at p=0.49 -> actual label is 0.
    # We want the majority color to be target_label; minority is 1-target_label.
    # Number of cells in circle: pi*r^2. For minority density m, need r^2 ≈ m*L^2/pi.
    # Let m = 0.49. So r = sqrt(0.49 * L^2 / pi) ~ 0.394 L.
    blob = np.full((L, L), target_label, dtype=np.uint8)
    minority = 1 - target_label
    cy, cx = L // 2, L // 2
    r = int(0.394 * L)
    mask = (ii - cy) ** 2 + (jj - cx) ** 2 <= r * r
    blob[mask] = minority
    # Ensure majority is actually target_label (it should be since r was chosen carefully)
    actual_density_1 = blob.mean()
    out["one_big_blob"] = blob.copy()

    # --- two big minority blobs ---
    two = np.full((L, L), target_label, dtype=np.uint8)
    r2 = int(0.30 * L)
    mask1 = (ii - L // 4) ** 2 + (jj - L // 4) ** 2 <= r2 * r2
    mask2 = (ii - 3 * L // 4) ** 2 + (jj - 3 * L // 4) ** 2 <= r2 * r2
    two[mask1 | mask2] = minority
    out["two_blobs"] = two.copy()

    # --- half-half vertical split, one cell tipping ---
    half = np.full((L, L), target_label, dtype=np.uint8)
    half[:, : L // 2] = minority
    # density = 0.5; flip to tip
    if target_label == 1:
        half[0, 0] = 1  # flip minority cell to majority
    else:
        half[0, L // 2] = 0
    out["half_half_vertical"] = half.copy()

    # --- near-critical random (single seed; minority density = 0.5 - 1/L^2) ---
    density = 0.5 - (1.0 / (L * L)) if target_label == 1 else 0.5 + (1.0 / (L * L))
    # We want majority = target_label, so for label=1 we want p(1) > 0.5
    if target_label == 1:
        p = 0.5 + 2.0 / (L * L)  # tiny majority of 1s
    else:
        p = 0.5 - 2.0 / (L * L)
    rc = (rng.random((L, L)) < p).astype(np.uint8)
    # Verify / adjust majority
    total1 = int(rc.sum())
    half = L * L // 2
    if target_label == 1 and total1 <= half:
        # flip enough cells from 0 to 1 to reach > half
        zeros_idx = np.nonzero(rc == 0)
        need = half - total1 + 1
        for k in range(need):
            rc[zeros_idx[0][k], zeros_idx[1][k]] = 1
    elif target_label == 0 and total1 >= half:
        ones_idx = np.nonzero(rc == 1)
        need = total1 - half + 1
        for k in range(need):
            rc[ones_idx[0][k], ones_idx[1][k]] = 0
    out["near_critical_random"] = rc.copy()

    return out


def run_schedule(
    catalog,
    sid_pre: str,
    sid_shake: str,
    *,
    T_pre: int,
    T_amp: int,
    T_shake: int,
    num_shakes: int,
    T_amp_final: int,
    initial_states: np.ndarray,
    labels: np.ndarray,
    backend,
) -> np.ndarray:
    idx_pre = catalog.resolve_rule_ref(sid_pre)
    F_pre = catalog.lut_bits[idx_pre].astype(np.uint8)
    idx_shake = catalog.resolve_rule_ref(sid_shake)
    F_shake = catalog.lut_bits[idx_shake].astype(np.uint8)

    batch, h, w = initial_states.shape
    L2 = h * w
    states = backend.asarray(initial_states, dtype="uint8")
    tiled_pre = backend.asarray(np.tile(F_pre, (batch, 1)), dtype="uint8")
    tiled_shake = backend.asarray(np.tile(F_shake, (batch, 1)), dtype="uint8")

    for _ in range(T_pre):
        states = backend.step_pairwise(states, tiled_pre)
    for _ in range(num_shakes):
        for _ in range(T_amp):
            states = apply_radius_step_mlx(states, "moore81")
        for _ in range(T_shake):
            states = backend.step_pairwise(states, tiled_shake)
    for _ in range(T_amp_final):
        states = apply_radius_step_mlx(states, "moore81")

    return backend.to_numpy(states)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", type=Path, default=_CATALOG_DIR / "expanded_property_panel_nonzero.json")
    parser.add_argument("--sid-pre", default="sid:58ed6b657afb")
    parser.add_argument("--sid-shake", default="sid:58ed6b657afb")
    parser.add_argument("--grids", type=int, nargs="+", default=[192, 256, 384])
    parser.add_argument("--T-pre", type=int, default=128)
    parser.add_argument("--T-amp", type=int, default=256)
    parser.add_argument("--T-shake", type=int, default=64)
    parser.add_argument("--num-shakes", type=int, default=2)
    parser.add_argument("--T-amp-final", type=int, default=1024)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument("--output", type=Path, default=Path("adversarial.json"))
    args = parser.parse_args()

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    backend = create_backend(args.backend)
    rng = np.random.default_rng(42)

    runs = []
    for grid in args.grids:
        # Build test batch: for each adversarial kind and each target_label,
        # we have one init state. So 2 labels x N kinds = 2N initial states.
        inits = []
        labels = []
        kinds = []
        for target_label in (0, 1):
            batch = make_adversarial_batch(grid, target_label, rng)
            for name, state in batch.items():
                tot = int(state.sum())
                half = grid * grid // 2
                actual_label = 1 if tot > half else 0
                inits.append(state)
                labels.append(actual_label)
                kinds.append(f"{name}_lbl{target_label}")
        initial_states = np.stack(inits, axis=0)
        labels_arr = np.asarray(labels, dtype=np.uint8)

        t0 = time.time()
        final = run_schedule(
            catalog, args.sid_pre, args.sid_shake,
            T_pre=args.T_pre, T_amp=args.T_amp,
            T_shake=args.T_shake, num_shakes=args.num_shakes,
            T_amp_final=args.T_amp_final,
            initial_states=initial_states,
            labels=labels_arr,
            backend=backend,
        )
        totals = final.reshape(len(inits), -1).sum(axis=1)
        all_zero = (totals == 0)
        all_one = (totals == grid * grid)
        consensus = all_zero | all_one
        correct = (all_zero & (labels_arr == 0)) | (all_one & (labels_arr == 1))

        per_case = []
        for k, name in enumerate(kinds):
            per_case.append({
                "kind": name,
                "label": int(labels_arr[k]),
                "initial_density": float(initial_states[k].mean()),
                "final_density": float(final[k].mean()),
                "consensus": bool(consensus[k]),
                "correct": bool(correct[k]),
            })
        runs.append({
            "grid": grid,
            "elapsed_seconds": time.time() - t0,
            "num_cases": len(kinds),
            "num_correct": int(correct.sum()),
            "num_consensus": int(consensus.sum()),
            "per_case": per_case,
        })
        print(f"grid={grid:>4d}  {int(correct.sum())}/{len(kinds)} correct consensus  "
              f"({int(consensus.sum())} reached consensus)")
        for case in per_case:
            tag = "OK" if case["correct"] else ("WRONG-CON" if case["consensus"] else "NO-CON")
            print(f"  {tag:<10} {case['kind']:<30} init_d={case['initial_density']:.6f} "
                  f"final_d={case['final_density']:.6f}")

    args.output.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
