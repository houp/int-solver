"""P1.1 mechanism probe.

Runs sid:58ed6b657afb alone (no amplifier) on random near-critical 2D tori
and records time-series diagnostics that characterize what the rule actually
does to the lattice: cluster statistics, spectral content, interface length,
and local-majority decodability.

Output: JSON file with per-checkpoint metrics for several (grid size, seed,
probability) combinations. Snapshots at selected checkpoints are also saved
as compressed .npz so they can be visualized later.
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


DEFAULT_SID = "sid:58ed6b657afb"
DEFAULT_BINARY = _CATALOG_DIR / "expanded_property_panel_nonzero.bin"
DEFAULT_METADATA = _CATALOG_DIR / "expanded_property_panel_nonzero.json"


def moore_majority_field(states: np.ndarray) -> np.ndarray:
    """Per-site Moore-majority (sum of 9-cell neighborhood >= 5)."""
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


def vn_majority_field(states: np.ndarray) -> np.ndarray:
    y = np.roll(states, 1, axis=1)
    t = np.roll(states, 1, axis=2)
    u = states
    w = np.roll(states, -1, axis=2)
    b = np.roll(states, -1, axis=1)
    s = (y + t + u + w + b).astype(np.int16)
    return (s >= 3).astype(np.uint8)


def interface_length(state: np.ndarray) -> int:
    """Total number of unequal horizontally+vertically adjacent pairs (torus)."""
    h = int((state != np.roll(state, -1, axis=0)).sum())
    v = int((state != np.roll(state, -1, axis=1)).sum())
    return h + v


def largest_component_size(state: np.ndarray, value: int) -> tuple[int, int]:
    """Return (num_components, largest_size) for cells equal to `value`.

    Uses 4-connectivity with periodic boundary conditions (approximated by
    not wrapping — should be fine for large grids; bias is small-boundary only).
    """
    mask = (state == value).astype(np.uint8)
    labeled, n = ndimage.label(mask)
    if n == 0:
        return 0, 0
    sizes = np.bincount(labeled.ravel())
    # sizes[0] is background count
    largest = int(sizes[1:].max()) if n >= 1 else 0
    return int(n), largest


def radial_spectrum(state: np.ndarray, max_k_bins: int = 16) -> list[float]:
    """Compute a coarse radial power spectrum of the centered state field.

    Returns mean squared magnitude in `max_k_bins` annuli of equal width in
    Fourier radius. Useful to track whether low-k modes (large structures)
    grow relative to high-k modes (noise).
    """
    h, w = state.shape
    centered = state.astype(np.float64) - state.mean()
    # Fourier transform with shift so low-k is at center
    fft = np.fft.fftshift(np.fft.fft2(centered))
    power = (fft.real ** 2 + fft.imag ** 2)
    # Normalize
    power /= (h * w)
    ys = np.arange(h) - h // 2
    xs = np.arange(w) - w // 2
    rr = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
    rmax = min(h, w) / 2.0
    bin_edges = np.linspace(0.0, rmax, max_k_bins + 1)
    out = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        sel = (rr >= lo) & (rr < hi)
        out.append(float(power[sel].mean()) if sel.any() else 0.0)
    return out


def per_checkpoint_metrics(
    states_np: np.ndarray, initial_labels: np.ndarray
) -> dict[str, list[float] | list[list[float]]]:
    """Compute per-trial metrics and aggregate across the batch.

    `states_np` shape: (batch, H, W). `initial_labels` shape: (batch,), values in {0,1}.
    """
    batch, h, w = states_np.shape
    moore = moore_majority_field(states_np)
    vn = vn_majority_field(states_np)

    densities: list[float] = []
    moore_agree: list[float] = []
    vn_agree: list[float] = []
    decodability_strict: list[float] = []  # fraction of sites where BOTH moore and VN local majority equal global majority
    interface: list[int] = []
    num_components_maj: list[int] = []
    largest_maj_fraction: list[float] = []
    density_variance: list[float] = []
    spectra: list[list[float]] = []

    for i in range(batch):
        s = states_np[i]
        label = int(initial_labels[i])
        densities.append(float(s.mean()))
        target = label
        moore_agree.append(float((moore[i] == target).mean()))
        vn_agree.append(float((vn[i] == target).mean()))
        decodability_strict.append(
            float(((moore[i] == target) & (vn[i] == target)).mean())
        )
        interface.append(interface_length(s))
        n, lg = largest_component_size(s, label)
        num_components_maj.append(n)
        largest_maj_fraction.append(lg / float(h * w))
        # local-density variance using Moore neighborhood mean
        mean_field = (
            s.astype(np.float32)
            + np.roll(s, 1, axis=0) + np.roll(s, -1, axis=0)
            + np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1)
            + np.roll(np.roll(s, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(s, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(s, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(s, -1, axis=0), -1, axis=1)
        ) / 9.0
        density_variance.append(float(mean_field.var()))
        spectra.append(radial_spectrum(s))

    return {
        "densities": densities,
        "moore_majority_agreement": moore_agree,
        "vn_majority_agreement": vn_agree,
        "decodability_strict": decodability_strict,
        "interface_length": interface,
        "num_majority_components": num_components_maj,
        "largest_majority_component_fraction": largest_maj_fraction,
        "local_density_variance": density_variance,
        "radial_spectrum": spectra,
    }


def run_probe(
    rule_bits: np.ndarray,
    *,
    grid: int,
    probability: float,
    trials: int,
    max_steps: int,
    checkpoints: list[int],
    seed: int,
    backend_name: str,
    snapshot_every: int | None,
    snapshot_dir: Path | None,
    snapshot_label: str,
) -> dict:
    rng = np.random.default_rng(seed)
    init = (rng.random((trials, grid, grid)) < probability).astype(np.uint8)
    totals = init.reshape(trials, -1).sum(axis=1)
    initial_labels = (totals > (grid * grid / 2)).astype(np.uint8)
    tied = (totals * 2 == grid * grid)

    backend = create_backend(backend_name)
    states = backend.asarray(init, dtype="uint8")
    tiled_rule = np.tile(rule_bits, (trials, 1)).astype(np.uint8)
    tiled_rule = backend.asarray(tiled_rule, dtype="uint8")

    results_by_step: dict[int, dict] = {}
    # Step 0 metrics
    snap0 = backend.to_numpy(states)
    results_by_step[0] = per_checkpoint_metrics(snap0, initial_labels)
    if snapshot_dir is not None:
        np.savez_compressed(snapshot_dir / f"{snapshot_label}_step00000.npz", snap0)

    checkpoint_set = set(checkpoints)
    current_step = 0
    t0 = time.time()
    while current_step < max_steps:
        target = current_step + 1
        # run one step
        states = backend.step_pairwise(states, tiled_rule)
        current_step = target
        if current_step in checkpoint_set:
            snap = backend.to_numpy(states)
            results_by_step[current_step] = per_checkpoint_metrics(snap, initial_labels)
            if snapshot_dir is not None and (
                snapshot_every is None or current_step % snapshot_every == 0
            ):
                np.savez_compressed(
                    snapshot_dir / f"{snapshot_label}_step{current_step:05d}.npz",
                    snap,
                )

    elapsed = time.time() - t0
    return {
        "grid": grid,
        "probability": probability,
        "trials": trials,
        "tied_initial_states": int(tied.sum()),
        "seed": seed,
        "backend": backend_name,
        "max_steps": max_steps,
        "elapsed_seconds": elapsed,
        "initial_label_counts": {
            "zero": int((initial_labels == 0).sum()),
            "one": int((initial_labels == 1).sum()),
        },
        "metrics_by_step": {str(k): v for k, v in sorted(results_by_step.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", default=DEFAULT_SID)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--backend", default="mlx")
    parser.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Grid sizes (square tori) to probe",
    )
    parser.add_argument(
        "--probabilities",
        type=float,
        nargs="+",
        default=[0.49, 0.51],
        help="Initial Bernoulli densities",
    )
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202604)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("mechanism_58ed6.json"))
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("mechanism_58ed6_snapshots"),
        help="Directory to save compressed lattice snapshots",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=None,
        help="If set, only save snapshots at checkpoints that are multiples of this",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Disable snapshot saving entirely",
    )
    args = parser.parse_args()

    snapshot_dir: Path | None = None
    if not args.no_snapshots:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = args.snapshot_dir

    catalog = load_binary_catalog(str(args.binary), str(args.metadata))
    idx = catalog.resolve_rule_ref(args.sid)
    rule_bits = catalog.lut_bits[idx].astype(np.uint8)
    resolved_sid = catalog.stable_ids[idx]

    checkpoints = sorted(
        {0}
        | {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
        | {args.max_steps}
    )
    checkpoints = [c for c in checkpoints if c <= args.max_steps]

    runs = []
    for grid in args.grids:
        for prob in args.probabilities:
            label = f"g{grid}_p{prob:.2f}"
            print(f"[probe] grid={grid} p={prob} trials={args.trials} max_steps={args.max_steps}")
            run = run_probe(
                rule_bits,
                grid=grid,
                probability=prob,
                trials=args.trials,
                max_steps=args.max_steps,
                checkpoints=checkpoints,
                seed=args.seed + 1000 * grid + int(prob * 1000),
                backend_name=args.backend,
                snapshot_every=args.snapshot_every,
                snapshot_dir=snapshot_dir,
                snapshot_label=label,
            )
            run["label"] = label
            runs.append(run)
            print(
                f"  done in {run['elapsed_seconds']:.1f}s; "
                f"first label counts {run['initial_label_counts']}"
            )

    out = {
        "rule_sid": resolved_sid,
        "rule_legacy_index": int(idx),
        "checkpoints": checkpoints,
        "runs": runs,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
