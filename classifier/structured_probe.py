"""Probe what each phase of the current schedule does to structured initial
states. The goal is to see at which phase the schedule fails to break
structural invariants.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path


# --- repo-root path bootstrap (keep BEFORE ca_search/local imports) ---
import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_REPO_ROOT))
# ---------------------------------------------------------------------

from amplifier_library import apply_radius_step_mlx
from ca_search.binary_catalog import load_binary_catalog
from ca_search.simulator import create_backend


def make_stripes(L: int, orientation: str, target_label: int) -> np.ndarray:
    s = np.zeros((L, L), dtype=np.uint8)
    if orientation == "h":
        s[::2] = 1
    else:
        s[:, ::2] = 1
    if target_label == 1:
        s[0, 0] = 1 if s[0, 0] == 0 else s[0, 0]
        if s.sum() * 2 <= L * L:
            # flip one 0 to 1
            idx = np.argwhere(s == 0)[0]
            s[idx[0], idx[1]] = 1
    else:
        if s.sum() * 2 >= L * L:
            idx = np.argwhere(s == 1)[0]
            s[idx[0], idx[1]] = 0
    return s


def make_checkerboard(L: int, target_label: int) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    s = ((ii + jj) % 2).astype(np.uint8)
    if target_label == 1 and s.sum() * 2 <= L * L:
        idx = np.argwhere(s == 0)[0]
        s[idx[0], idx[1]] = 1
    elif target_label == 0 and s.sum() * 2 >= L * L:
        idx = np.argwhere(s == 1)[0]
        s[idx[0], idx[1]] = 0
    return s


def make_half_half(L: int, target_label: int) -> np.ndarray:
    s = np.full((L, L), target_label, dtype=np.uint8)
    other = 1 - target_label
    s[:, : L // 2] = other
    # flip one cell to tip
    if target_label == 1 and s.sum() * 2 <= L * L:
        s[0, 0] = 1
    elif target_label == 0 and s.sum() * 2 >= L * L:
        s[0, L // 2] = 0
    return s


def describe(state: np.ndarray) -> str:
    h, w = state.shape
    d = float(state.mean())
    ih = int((state != np.roll(state, -1, axis=0)).sum())
    iv = int((state != np.roll(state, -1, axis=1)).sum())
    return f"d={d:.4f} iface_h={ih} iface_v={iv}"


def apply_rule(states, rule_bits, backend, steps):
    batch = states.shape[0] if states.ndim == 3 else 1
    if states.ndim == 2:
        states = states[None]
    backend_state = backend.asarray(states, dtype="uint8")
    tiled = backend.asarray(np.tile(rule_bits, (batch, 1)), dtype="uint8")
    for _ in range(steps):
        backend_state = backend.step_pairwise(backend_state, tiled)
    return backend.to_numpy(backend_state)


def apply_moore81(states, backend, steps):
    if states.ndim == 2:
        states = states[None]
    bstate = backend.asarray(states, dtype="uint8")
    for _ in range(steps):
        bstate = apply_radius_step_mlx(bstate, "moore81")
    return backend.to_numpy(bstate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--sid-pre", default="sid:58ed6b657afb")
    parser.add_argument("--T-pre", type=int, default=128)
    parser.add_argument("--T-amp", type=int, default=256)
    parser.add_argument("--T-shake", type=int, default=64)
    parser.add_argument("--num-shakes", type=int, default=2)
    parser.add_argument("--T-amp-final", type=int, default=1024)
    parser.add_argument("--target-label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--binary", default="expanded_property_panel_nonzero.bin")
    parser.add_argument("--metadata", default="expanded_property_panel_nonzero.json")
    args = parser.parse_args()

    catalog = load_binary_catalog(args.binary, args.metadata)
    idx = catalog.resolve_rule_ref(args.sid_pre)
    F_bits = catalog.lut_bits[idx].astype(np.uint8)
    backend = create_backend("mlx")

    cases = {
        "stripes_h": make_stripes(args.L, "h", args.target_label),
        "stripes_v": make_stripes(args.L, "v", args.target_label),
        "checker": make_checkerboard(args.L, args.target_label),
        "half_half": make_half_half(args.L, args.target_label),
    }

    print(f"L={args.L}  target_label={args.target_label}")
    print(f"sid={args.sid_pre[:20]}  T_pre={args.T_pre}")
    print()

    for name, init in cases.items():
        print(f"=== {name} ===")
        print(f"  init:           {describe(init)}")
        after1 = apply_rule(init, F_bits, backend, args.T_pre)[0]
        print(f"  after F^{args.T_pre}:     {describe(after1)}")
        after_amp1 = apply_moore81(after1, backend, args.T_amp)[0]
        print(f"  after M^{args.T_amp}:     {describe(after_amp1)}")
        after_shake1 = apply_rule(after_amp1, F_bits, backend, args.T_shake)[0]
        print(f"  after F^{args.T_shake}:      {describe(after_shake1)}")
        after_amp2 = apply_moore81(after_shake1, backend, args.T_amp)[0]
        print(f"  after M^{args.T_amp}:     {describe(after_amp2)}")
        after_shake2 = apply_rule(after_amp2, F_bits, backend, args.T_shake)[0]
        print(f"  after F^{args.T_shake}:      {describe(after_shake2)}")
        final = apply_moore81(after_shake2, backend, args.T_amp_final)[0]
        print(f"  after M^{args.T_amp_final}:    {describe(final)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
