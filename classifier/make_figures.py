"""Generate figures for the technical report's density-classification section.

Reads the committed result JSONs and produces vector PDFs in docs/figures/.

Figures generated:
1. rho_min_scaling.pdf  -- log-log scaling of rho_min(L), random and
                           adversarial, with least-squares fit lines.
2. head_to_head_steps.pdf -- bar chart of step count per trial,
                              ours vs Fates-Regnault, log scale.
3. snapshots_random.pdf -- 4-panel: initial random Bernoulli, after F^{T_pre},
                           after [moore81 F swap]^2, after moore81^{T_amp_final}
                           -- showing lattice evolution toward consensus.
4. snapshots_adversarial_stripes.pdf -- same 4-panel for a stripe input.
5. snapshots_fates_regnault.pdf -- 3-panel: initial, after Fates Phase 1
                                    (F_X o R_184)^{T_1}, after Phase 2
                                    -- showing the FR construction at work.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; required for PDF without a display
import matplotlib.pyplot as plt


# --- repo-root path bootstrap ---
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
_CATALOG_DIR = _REPO_ROOT / "catalogs"
# --------------------------------

from amplifier_library import apply_radius_step_mlx
from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend
from conservative_noise import apply_lut_mlx, apply_swaps
from adversarial_realistic import make_stripe_with_density
from fates_regnault_2016 import (
    step_F_X, step_R_184_2d, step_G_X, step_R_232_2d,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_rho_min_scaling(*, sweep_paths: list[Path], out: Path) -> None:
    """Log-log plot of rho_min(L) for both datasets, with least-squares fits."""
    runs: list[dict] = []
    for p in sweep_paths:
        if not p.exists():
            continue
        runs.extend(json.loads(p.read_text())["runs"])
    if not runs:
        print(f"  no sweep data; skipping {out}")
        return
    # dedup by (L, delta), latest wins
    dedup = {}
    for r in runs:
        dedup[(r["grid"], r["delta"])] = r
    runs = list(dedup.values())

    grids = sorted({r["grid"] for r in runs})
    deltas = sorted({r["delta"] for r in runs}, reverse=True)

    rho_min = {"random": {}, "adversarial": {}, "joint": {}}
    for L in grids:
        for d in sorted(deltas, reverse=True):
            r = next((r for r in runs if r["grid"] == L and r["delta"] == d), None)
            if r is None:
                continue
            if r["random_cc"] == 1.0:
                rho_min["random"][L] = d
            if r["adversarial_cc"] == 1.0:
                rho_min["adversarial"][L] = d
            if r["random_cc"] == 1.0 and r["adversarial_cc"] == 1.0:
                rho_min["joint"][L] = d

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    series = [
        ("random", "o-", "tab:blue"),
        ("adversarial", "s-", "tab:red"),
    ]
    for label, marker, color in series:
        Ls = sorted(rho_min[label].keys())
        if len(Ls) < 2:
            continue
        ys = [rho_min[label][L] for L in Ls]
        ax.loglog(Ls, ys, marker, color=color, label=f"{label} data", markersize=6)
        # fit
        xs = np.log(Ls); ysl = np.log(ys)
        A = np.vstack([xs, np.ones_like(xs)]).T
        slope, intercept = np.linalg.lstsq(A, ysl, rcond=None)[0]
        x_fit = np.linspace(min(Ls) * 0.9, max(Ls) * 1.1, 50)
        y_fit = math.exp(intercept) * x_fit ** slope
        ax.loglog(x_fit, y_fit, "--", color=color, alpha=0.6,
                  label=f"  fit  $\\propto L^{{{slope:.2f}}}$")
    # Reference line: 1/L^2 (information-theoretic floor)
    L_ref = np.array(grids, dtype=float)
    if len(L_ref) >= 2:
        ax.loglog(L_ref, 1.0 / L_ref ** 2, ":", color="gray",
                  label=r"per-cell floor $L^{-2}$")
    ax.set_xlabel(r"grid size $L$")
    ax.set_ylabel(r"$\rho_{\min}(L)$  (smallest $\delta$ giving $100\%$)")
    ax.set_title(r"Empirical density-margin floor of the L-scaled stochastic classifier")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, out)


def fig_head_to_head_steps(*, h2h_path: Path, out: Path) -> None:
    """Steps-per-trial comparison: ours vs FR, log y-scale."""
    if not h2h_path.exists():
        print(f"  {h2h_path} missing; skipping {out}")
        return
    data = json.loads(h2h_path.read_text())
    runs = data["runs"]
    grids = sorted({r["L"] for r in runs})

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ours_steps = [next((r for r in runs if r["L"] == L), None)["ours_steps_per_trial"] for L in grids]
    fr_steps = [next((r for r in runs if r["L"] == L), None)["fr_steps_per_trial"] for L in grids]
    width = 0.38
    xs = np.arange(len(grids))
    ax.bar(xs - width / 2, ours_steps, width, color="tab:blue", label="our schedule")
    ax.bar(xs + width / 2, fr_steps, width, color="tab:red", label=r"Fat\`{e}s--Regnault")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(grids)
    ax.set_xlabel(r"grid size $L$")
    ax.set_ylabel("CA steps per trial")
    ax.set_title("Schedule cost per trial (log scale)")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.25)
    _save(fig, out)


def fig_head_to_head_correctness(*, h2h_path: Path, out: Path) -> None:
    """Grouped bar chart of correct-consensus rates per (L, delta) for the
    adversarial battery, ours vs FR."""
    if not h2h_path.exists():
        print(f"  {h2h_path} missing; skipping {out}")
        return
    data = json.loads(h2h_path.read_text())
    runs = data["runs"]
    rows = []
    for r in runs:
        rows.append((r["L"], r["delta"], r["ours_adv_cc"], r["fr_adv_cc"]))
    rows.sort()
    labels = [f"L={L}\n δ={d:.3f}" for (L, d, _, _) in rows]
    ours = [c for (_, _, c, _) in rows]
    fr = [c for (_, _, _, c) in rows]
    width = 0.38
    xs = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(xs - width / 2, ours, width, color="tab:blue", label="our schedule")
    ax.bar(xs + width / 2, fr, width, color="tab:red", label=r"Fat\`{e}s--Regnault")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("correct-consensus rate (adversarial)")
    ax.set_title(r"Adversarial robustness: ours vs Fat\`{e}s--Regnault on identical inputs")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    _save(fig, out)


def _imshow_panel(ax, lattice, title, cmap="Greys"):
    ax.imshow(lattice, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def fig_snapshots_ours(*, L: int, density: float, c_pre: float, c_amp: float,
                         c_shake: float, c_final: float, k_swap: float,
                         num_shakes: int, F_bits: np.ndarray, backend,
                         seed: int, sched_label: str, init_kind: str,
                         out: Path, init_lattice: np.ndarray | None = None) -> None:
    """4-panel snapshot of our schedule on either a random or stripe input."""
    if init_lattice is None:
        rng_init = np.random.default_rng(seed)
        if init_kind == "random":
            init_total = int(round(density * L * L))
            flat = np.zeros(L * L, dtype=np.uint8); flat[:init_total] = 1
            rng_init.shuffle(flat)
            init = flat.reshape(L, L)
        else:
            init = make_stripe_with_density(L, "h", density, rng_init)
    else:
        init = init_lattice
    init_state = init[None]

    T_pre = int(c_pre * L); T_amp = int(c_amp * L)
    T_shake = int(c_shake * L); T_amp_final = int(c_final * L)
    K = int(k_swap * L)
    rng = np.random.default_rng(seed + 1)
    states = backend.asarray(init_state, dtype="uint8")
    tiled_F = backend.asarray(np.tile(F_bits, (1, 1)), dtype="uint8")

    snap0 = init_state[0].copy()
    for _ in range(T_pre):
        states = apply_lut_mlx(states, tiled_F)
    snap1 = backend.to_numpy(states)[0].copy()
    for _ in range(num_shakes):
        for _ in range(T_amp):
            states = apply_radius_step_mlx(states, "moore81")
        for _ in range(T_shake):
            states = apply_lut_mlx(states, tiled_F)
        s_np = backend.to_numpy(states)
        s_np = apply_swaps(s_np, K, rng)
        states = backend.asarray(s_np, dtype="uint8")
    snap2 = backend.to_numpy(states)[0].copy()
    for _ in range(T_amp_final):
        states = apply_radius_step_mlx(states, "moore81")
    snap3 = backend.to_numpy(states)[0].copy()

    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.2))
    _imshow_panel(axes[0], snap0, f"initial ({init_kind})\n$\\rho={snap0.mean():.4f}$")
    _imshow_panel(axes[1], snap1, f"after $F^{{{T_pre}}}$\n$\\rho={snap1.mean():.4f}$")
    _imshow_panel(axes[2], snap2, f"after shake cycles\n$\\rho={snap2.mean():.4f}$")
    _imshow_panel(axes[3], snap3, f"after $M^{{{T_amp_final}}}$\n$\\rho={snap3.mean():.4f}$")
    fig.suptitle(sched_label, fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def fig_snapshots_fates(*, L: int, density: float, T1: int, T2: int,
                          seed: int, init_kind: str, out: Path) -> None:
    """3-panel snapshot of the Fates-Regnault construction."""
    rng_init = np.random.default_rng(seed)
    if init_kind == "random":
        flat = np.zeros(L * L, dtype=np.uint8)
        flat[:int(round(density * L * L))] = 1
        rng_init.shuffle(flat)
        init = flat.reshape(L, L)
    else:
        init = make_stripe_with_density(L, "h", density, rng_init)
    snap0 = init.copy()
    s = init[None].copy()
    rng = np.random.default_rng(seed + 1)
    for _ in range(T1):
        X = (rng.random(s.shape) < 0.5).astype(np.uint8)
        s = step_F_X(s, X)
        s = step_R_184_2d(s)
    snap1 = s[0].copy()
    for _ in range(T2):
        X = (rng.random(s.shape) < 0.5).astype(np.uint8)
        s = step_G_X(s, X)
        s = step_R_232_2d(s)
    snap2 = s[0].copy()
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
    _imshow_panel(axes[0], snap0, f"initial ({init_kind})\n$\\rho={snap0.mean():.4f}$")
    _imshow_panel(axes[1], snap1,
                    rf"after $(F_X\,R_{{184}})^{{{T1}}}$" + f"\n$\\rho={snap1.mean():.4f}$")
    _imshow_panel(axes[2], snap2,
                    rf"after $(G_X\,R_{{232}})^{{{T2}}}$" + f"\n$\\rho={snap2.mean():.4f}$")
    fig.suptitle(rf"Fat\`{{e}}s--Regnault classifier at $L={L}$", fontsize=10)
    fig.tight_layout()
    _save(fig, out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "docs" / "figures")
    parser.add_argument("--sid", default="sid:58ed6b657afb")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snapshot-L", type=int, default=128)
    parser.add_argument("--snapshot-fr-L", type=int, default=50)
    parser.add_argument("--sweep-paths", type=Path, nargs="+",
                        default=[
                            _REPO_ROOT / "results" / "density_classification" / "2026-04-27" / "density_margin_sweep.json",
                            _REPO_ROOT / "results" / "density_classification" / "2026-04-27" / "density_margin_sweep_large.json",
                        ])
    parser.add_argument("--h2h-path", type=Path,
                        default=_REPO_ROOT / "results" / "density_classification" / "2026-04-27" / "head_to_head_fr.json")
    parser.add_argument("--skip-snapshots", action="store_true")
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--skip-h2h", action="store_true")
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_scaling:
        fig_rho_min_scaling(sweep_paths=args.sweep_paths, out=out / "rho_min_scaling.pdf")

    if not args.skip_h2h:
        fig_head_to_head_steps(h2h_path=args.h2h_path, out=out / "head_to_head_steps.pdf")
        fig_head_to_head_correctness(h2h_path=args.h2h_path, out=out / "head_to_head_correctness.pdf")

    if not args.skip_snapshots:
        catalog = load_binary_catalog(
            str(_CATALOG_DIR / "expanded_property_panel_nonzero.bin"),
            str(_CATALOG_DIR / "expanded_property_panel_nonzero.json"))
        F_bits = catalog.lut_bits[catalog.resolve_rule_ref(args.sid)].astype(np.uint8)
        backend = create_backend("mlx")

        L = args.snapshot_L
        common = dict(
            L=L, c_pre=0.5, c_amp=1.0, c_shake=0.25, c_final=4.0,
            k_swap=8.0, num_shakes=2, F_bits=F_bits, backend=backend,
        )
        fig_snapshots_ours(
            **common,
            density=0.51,
            seed=args.seed,
            sched_label=rf"Our scaled schedule on a near-critical random input ($L={L}$, $\rho=0.51$)",
            init_kind="random",
            out=out / "snapshots_ours_random.pdf",
        )
        fig_snapshots_ours(
            **common,
            density=0.55,
            seed=args.seed + 1,
            sched_label=rf"Our scaled schedule on a stripe-density input ($L={L}$, $\rho=0.55$)",
            init_kind="stripes",
            out=out / "snapshots_ours_stripes.pdf",
        )

        L_fr = args.snapshot_fr_L
        T_fr = 3 * L_fr * L_fr
        fig_snapshots_fates(
            L=L_fr, density=0.51, T1=T_fr, T2=T_fr,
            seed=args.seed, init_kind="random",
            out=out / "snapshots_fates_random.pdf",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
