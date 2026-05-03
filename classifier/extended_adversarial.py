"""Extended adversarial input families for stricter benchmarking.

The original adversarial_realistic battery has 5 pattern families
(stripes_h, stripes_v, checker, block_checker, half_half).  Here we
add three more pattern families designed to test different failure
modes of the strict-CA mixture:

- ``concentric_rings``: alternating-color radial bands centered at the
  lattice center.  Tests rotation-invariant structures that none of
  the cardinal/diagonal traffic rules align with.
- ``voronoi``: K-seed Voronoi tessellation, each cell coloured by
  parity of nearest-seed index.  Tests irregular-blocky structure.
- ``fourier_mode``: low-frequency sinusoidal pattern thresholded to a
  binary lattice.  Tests soft-gradient structures rather than crisp
  boundaries.

All families produce lattices at exact target density via the same
``with_density`` adjustment loop used by adversarial_realistic.
Together with the original 5, the expanded battery has 8 pattern
families x 2 labels x 3 density margins = 48 cases.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _density_adjust_inplace(s: np.ndarray, density: float, rng: np.random.Generator) -> np.ndarray:
    """Mutate s so that its density equals exactly round(density * L^2) / L^2."""
    L_sq = s.size
    target_ones = int(round(density * L_sq))
    cur_ones = int(s.sum())
    diff = target_ones - cur_ones
    if diff > 0:
        z = np.argwhere(s == 0)
        sel = rng.choice(len(z), size=diff, replace=False)
        for idx in sel:
            r, c = z[idx]
            s[r, c] = 1
    elif diff < 0:
        o = np.argwhere(s == 1)
        sel = rng.choice(len(o), size=-diff, replace=False)
        for idx in sel:
            r, c = o[idx]
            s[r, c] = 0
    return s


def make_concentric_rings_with_density(L: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Alternating-colour radial bands at the lattice centre, then density-
    adjusted by random flips."""
    cy, cx = (L - 1) / 2.0, (L - 1) / 2.0
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    r = np.sqrt((ii - cy) ** 2 + (jj - cx) ** 2)
    band_width = max(1.0, L / 16.0)
    s = ((r // band_width).astype(np.int64) % 2).astype(np.uint8)
    return _density_adjust_inplace(s, density, rng)


def make_voronoi_with_density(L: int, density: float, rng: np.random.Generator,
                                n_seeds: int | None = None) -> np.ndarray:
    """Voronoi tessellation: pick n_seeds random sites, colour each cell by
    the parity of the index of its nearest site."""
    if n_seeds is None:
        n_seeds = max(8, L // 8)
    seed_y = rng.integers(0, L, size=n_seeds)
    seed_x = rng.integers(0, L, size=n_seeds)
    parity = np.arange(n_seeds, dtype=np.int64) % 2

    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    # Toroidal-aware nearest-site assignment: for each cell, compute distance
    # to each seed using the wrap-aware metric, take argmin.
    best_d2 = np.full((L, L), np.inf)
    best_idx = np.zeros((L, L), dtype=np.int64)
    for k in range(n_seeds):
        dy = np.abs(ii - seed_y[k]); dy = np.minimum(dy, L - dy)
        dx = np.abs(jj - seed_x[k]); dx = np.minimum(dx, L - dx)
        d2 = dy * dy + dx * dx
        mask = d2 < best_d2
        best_d2 = np.where(mask, d2, best_d2)
        best_idx = np.where(mask, k, best_idx)
    s = parity[best_idx].astype(np.uint8)
    return _density_adjust_inplace(s, density, rng)


def make_fourier_mode_with_density(L: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency sinusoidal pattern thresholded to binary."""
    # Pick a random low-frequency mode in {1, 2, 3} per axis.
    kx = int(rng.integers(1, 4))
    ky = int(rng.integers(1, 4))
    phase_x = float(rng.random() * 2 * np.pi)
    phase_y = float(rng.random() * 2 * np.pi)
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    field = (
        np.sin(2 * np.pi * ky * ii / L + phase_y)
        + np.sin(2 * np.pi * kx * jj / L + phase_x)
    )
    # Threshold at 0 to get a roughly density-0.5 pattern.
    s = (field > 0).astype(np.uint8)
    return _density_adjust_inplace(s, density, rng)


def build_extended_adversarial_inits(
    L: int,
    densities: list[float],
    rng: np.random.Generator,
):
    """Build the 8-family x 2-label x len(densities)-density adversarial
    battery.  Returns (inits, labels, names).

    Inits and labels are numpy arrays; names is a list of strings of the
    form ``<family>_d<density>``.

    The 5 original families come from adversarial_realistic; the 3 new
    ones are defined in this module.
    """
    # Late import to avoid a hard cycle if both files are imported.
    from adversarial_realistic import (
        make_stripe_with_density,
        make_checker_with_density,
        make_block_checker_with_density,
        make_half_half_with_density,
    )

    cases, names, labels = [], [], []
    for d in densities:
        for label_bit in (0, 1):
            actual_d = d if label_bit == 1 else 1.0 - d
            target_label = 1 if actual_d > 0.5 else 0
            for name, fn in [
                ("stripes_h", lambda r=rng, dd=actual_d: make_stripe_with_density(L, "h", dd, r)),
                ("stripes_v", lambda r=rng, dd=actual_d: make_stripe_with_density(L, "v", dd, r)),
                ("checker", lambda r=rng, dd=actual_d: make_checker_with_density(L, dd, r)),
                ("block_checker", lambda r=rng, dd=actual_d: make_block_checker_with_density(L, dd, r)),
                ("half_half", lambda r=rng, dd=actual_d: make_half_half_with_density(L, dd, r)),
                ("rings", lambda r=rng, dd=actual_d: make_concentric_rings_with_density(L, dd, r)),
                ("voronoi", lambda r=rng, dd=actual_d: make_voronoi_with_density(L, dd, r)),
                ("fourier", lambda r=rng, dd=actual_d: make_fourier_mode_with_density(L, dd, r)),
            ]:
                s = fn()
                cases.append(s.copy())
                names.append(f"{name}_d{actual_d:.4f}")
                labels.append(target_label)
    return np.stack(cases), np.asarray(labels, dtype=np.uint8), names


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(0)
    inits, labels, names = build_extended_adversarial_inits(64, [0.51, 0.52, 0.55], rng)
    print(f"battery shape: {inits.shape}, labels: {labels.shape}")
    print(f"families: {sorted({n.rsplit('_d', 1)[0] for n in names})}")
    print(f"total cases: {len(names)} (expected 8 * 2 * 3 = 48)")
    # Verify exact density per case
    L_sq = 64 * 64
    for i in range(min(8, len(names))):
        n_ones = int(inits[i].sum())
        d = n_ones / L_sq
        print(f"  {names[i]:<22}  ones={n_ones:>5}  density={d:.4f}  label={labels[i]}")
