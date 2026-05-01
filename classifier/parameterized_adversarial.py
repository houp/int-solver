"""Parameterized adversarial input families.

The static eight-family battery in extended_adversarial.py saturates:
the recommended strict-CA mixture scores 100% on every cell across our
two-tier head-to-head benchmark.  To drill deeper we generate a much
larger set of structured inputs by parameterizing each family across
period, scale, angle, frequency, and combination axes.

Available families and their parameters:

- ``stripes_h_period_p``: horizontal alternating bands of period p
  (p in {1, 2, 3, 4, 5, 6, 8, 12}).  p=2 reproduces the original
  battery.
- ``stripes_v_period_p``: vertical version.
- ``stripes_diag_period_p``: 45-degree diagonal stripes of period p.
- ``checker_block_b``: block-checkerboard with b x b cells of one color
  (b in {1, 2, 3, 4, 5, 6, 8}).  b=1 is the standard checkerboard;
  b=2 is the original block_checker.
- ``rings_width_w``: concentric rings of width w (w in {1, 2, 3, 4, 6, 8}).
- ``half_split_axis_a``: bichromatic split along axis ``a`` in
  {h, v, diag, antidiag}.
- ``fourier_kx_ky``: low-frequency thresholded sinusoid with mode
  numbers (kx, ky) in {1, 2, 3, 4} x {1, 2, 3, 4}.
- ``voronoi_seeds_n``: Voronoi tessellation with n random seeds (n in
  {4, 8, 16, 32, 64}).
- ``nested_stripes_p1_p2``: outer stripe of period p1, inner stripe of
  period p2 within each band.
- ``noisy_stripes_p_eps``: stripe of period p plus eps-fraction random
  flips.

After generation each lattice is density-adjusted to exactly the target
density via the same "flip the right cells" loop used by the existing
families.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# --- repo-root path bootstrap ---
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
# --------------------------------

from extended_adversarial import _density_adjust_inplace


def make_stripes_period_h(L: int, period: int, density: float, rng: np.random.Generator) -> np.ndarray:
    s = np.zeros((L, L), dtype=np.uint8)
    for r in range(L):
        if (r // period) % 2 == 1:
            s[r, :] = 1
    return _density_adjust_inplace(s, density, rng)


def make_stripes_period_v(L: int, period: int, density: float, rng: np.random.Generator) -> np.ndarray:
    s = np.zeros((L, L), dtype=np.uint8)
    for c in range(L):
        if (c // period) % 2 == 1:
            s[:, c] = 1
    return _density_adjust_inplace(s, density, rng)


def make_stripes_diag(L: int, period: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """45-degree diagonal stripes of band width = period."""
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    s = (((ii + jj) // period) % 2).astype(np.uint8)
    return _density_adjust_inplace(s, density, rng)


def make_checker_block(L: int, block: int, density: float, rng: np.random.Generator) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    s = (((ii // block) + (jj // block)) % 2).astype(np.uint8)
    return _density_adjust_inplace(s, density, rng)


def make_rings_width(L: int, width: int, density: float, rng: np.random.Generator) -> np.ndarray:
    cy, cx = (L - 1) / 2.0, (L - 1) / 2.0
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    r = np.sqrt((ii - cy) ** 2 + (jj - cx) ** 2)
    s = ((r / width).astype(np.int64) % 2).astype(np.uint8)
    return _density_adjust_inplace(s, density, rng)


def make_half_split(L: int, axis: str, density: float, rng: np.random.Generator) -> np.ndarray:
    s = np.zeros((L, L), dtype=np.uint8)
    if axis == "h":
        s[L // 2 :, :] = 1
    elif axis == "v":
        s[:, L // 2 :] = 1
    elif axis == "diag":
        ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        s = (ii + jj >= L).astype(np.uint8)
    elif axis == "antidiag":
        ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        s = (ii > jj).astype(np.uint8)
    else:
        raise ValueError(f"unknown axis: {axis}")
    return _density_adjust_inplace(s, density, rng)


def make_fourier_mode(L: int, kx: int, ky: int, density: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Low-frequency sin(kx*x) + sin(ky*y), thresholded at zero."""
    phase_x = float(rng.random() * 2 * np.pi)
    phase_y = float(rng.random() * 2 * np.pi)
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    field = (
        np.sin(2 * np.pi * ky * ii / L + phase_y)
        + np.sin(2 * np.pi * kx * jj / L + phase_x)
    )
    s = (field > 0).astype(np.uint8)
    return _density_adjust_inplace(s, density, rng)


def make_voronoi_seeds(L: int, n_seeds: int, density: float, rng: np.random.Generator) -> np.ndarray:
    seed_y = rng.integers(0, L, size=n_seeds)
    seed_x = rng.integers(0, L, size=n_seeds)
    parity = np.arange(n_seeds, dtype=np.int64) % 2

    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
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


def make_nested_stripes(L: int, p_outer: int, p_inner: int, density: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Outer horizontal band of period p_outer; within each "1" band,
    apply vertical stripes of period p_inner."""
    s = np.zeros((L, L), dtype=np.uint8)
    for r in range(L):
        if (r // p_outer) % 2 == 1:
            for c in range(L):
                if (c // p_inner) % 2 == 1:
                    s[r, c] = 1
                else:
                    s[r, c] = 1  # entirely 1 in outer-1 band
            # Actually use inner only inside outer-1 bands:
            for c in range(L):
                s[r, c] = 1 if (c // p_inner) % 2 == 1 else 0
    return _density_adjust_inplace(s, density, rng)


def make_noisy_stripes(L: int, period: int, eps: float, density: float,
                        rng: np.random.Generator) -> np.ndarray:
    base = make_stripes_period_h(L, period, 0.5, rng)
    n_flip = int(round(eps * L * L))
    if n_flip > 0:
        flat = base.ravel()
        idx = rng.choice(L * L, size=n_flip, replace=False)
        flat[idx] = 1 - flat[idx]
        base = flat.reshape(L, L)
    return _density_adjust_inplace(base, density, rng)


def build_parameterized_battery(
    L: int,
    densities: list[float],
    rng: np.random.Generator,
    *,
    include_voronoi: bool = True,
    include_fourier: bool = True,
    include_nested: bool = True,
    include_noisy: bool = True,
):
    """Build the full parameterized adversarial battery and return
    (inits, labels, names) numpy arrays + list."""
    cases, names, labels = [], [], []

    # Adapt parameter ranges to L.
    period_choices = [p for p in [1, 2, 3, 4, 5, 6, 8, 12, 16] if p < L // 2]
    block_choices = [b for b in [1, 2, 3, 4, 5, 6, 8] if b < L // 4]
    ring_widths = [w for w in [1, 2, 3, 4, 6, 8] if w < L // 4]
    voronoi_seeds = [n for n in [4, 8, 16, 32, 64] if n < L * L // 4]
    fourier_modes = [(kx, ky) for kx in [1, 2, 3, 4] for ky in [1, 2, 3, 4] if kx + ky <= 5]
    half_axes = ["h", "v", "diag", "antidiag"]
    nested_pairs = [(p1, p2) for p1 in [4, 8] for p2 in [1, 2, 3] if p1 < L // 2]
    noise_levels = [0.05, 0.10, 0.20]
    noisy_periods = [2, 4, 8]

    for density_offset in densities:
        for label_bit in (0, 1):
            actual_d = density_offset if label_bit == 1 else 1.0 - density_offset
            target_label = 1 if actual_d > 0.5 else 0

            for p in period_choices:
                cases.append(make_stripes_period_h(L, p, actual_d, rng))
                names.append(f"stripes_h_p{p}_d{actual_d:.4f}")
                labels.append(target_label)
                cases.append(make_stripes_period_v(L, p, actual_d, rng))
                names.append(f"stripes_v_p{p}_d{actual_d:.4f}")
                labels.append(target_label)
                cases.append(make_stripes_diag(L, p, actual_d, rng))
                names.append(f"stripes_diag_p{p}_d{actual_d:.4f}")
                labels.append(target_label)
            for b in block_choices:
                cases.append(make_checker_block(L, b, actual_d, rng))
                names.append(f"checker_block{b}_d{actual_d:.4f}")
                labels.append(target_label)
            if include_fourier:
                for kx, ky in fourier_modes:
                    cases.append(make_fourier_mode(L, kx, ky, actual_d, rng))
                    names.append(f"fourier_kx{kx}_ky{ky}_d{actual_d:.4f}")
                    labels.append(target_label)
            for w in ring_widths:
                cases.append(make_rings_width(L, w, actual_d, rng))
                names.append(f"rings_w{w}_d{actual_d:.4f}")
                labels.append(target_label)
            for ax in half_axes:
                cases.append(make_half_split(L, ax, actual_d, rng))
                names.append(f"half_{ax}_d{actual_d:.4f}")
                labels.append(target_label)
            if include_voronoi:
                for n in voronoi_seeds:
                    cases.append(make_voronoi_seeds(L, n, actual_d, rng))
                    names.append(f"voronoi_n{n}_d{actual_d:.4f}")
                    labels.append(target_label)
            if include_nested:
                for p1, p2 in nested_pairs:
                    cases.append(make_nested_stripes(L, p1, p2, actual_d, rng))
                    names.append(f"nested_p1_{p1}_p2_{p2}_d{actual_d:.4f}")
                    labels.append(target_label)
            if include_noisy:
                for p in noisy_periods:
                    for eps in noise_levels:
                        cases.append(make_noisy_stripes(L, p, eps, actual_d, rng))
                        names.append(f"noisy_p{p}_e{eps:.2f}_d{actual_d:.4f}")
                        labels.append(target_label)

    return np.stack(cases), np.asarray(labels, dtype=np.uint8), names


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    inits, labels, names = build_parameterized_battery(64, [0.51, 0.52, 0.55], rng)
    print(f"battery shape: {inits.shape}, labels: {labels.shape}")
    print(f"family count: {len({n.rsplit('_d', 1)[0] for n in names})}")
    print(f"total cases: {len(names)}")
    families = {}
    for n in names:
        prefix = n.split("_p")[0].split("_b")[0].split("_n")[0].split("_w")[0].split("_e")[0].rstrip("_")
        families[prefix] = families.get(prefix, 0) + 1
    print("by family:")
    for f, c in sorted(families.items()):
        print(f"  {f}: {c}")
