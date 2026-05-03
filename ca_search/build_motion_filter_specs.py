from __future__ import annotations

from pathlib import Path

from .lut import POSITION_ORDER, VELOCITY_TO_SOURCE_POSITION


ROOT = Path(__file__).resolve().parents[1]
IDENTITIES_DIR = ROOT / "identities"


def _single_one_args(position: str) -> str:
    return ", ".join("1" if current == position else "0" for current in POSITION_ORDER)


def _single_zero_args(position: str) -> str:
    return ", ".join("0" if current == position else "1" for current in POSITION_ORDER)


def _render_particle_spec(velocity: str) -> str:
    target_position = VELOCITY_TO_SOURCE_POSITION[velocity]
    lines = [
        "vars: x, y, z, t, u, w, a, b, c",
        "",
        f"# Isolated-particle one-step motion: {velocity}.",
        "# The support of the single live cell is fully resolved over the affected 3x3 window.",
        "",
        "f(0, 0, 0, 0, 0, 0, 0, 0, 0) = 0;",
        "f(1, 1, 1, 1, 1, 1, 1, 1, 1) = 1;",
    ]
    for position in POSITION_ORDER:
        value = "1" if position == target_position else "0"
        lines.append(f"f({_single_one_args(position)}) = {value};")
    return "\n".join(lines) + "\n"


def _render_hole_spec(velocity: str) -> str:
    target_position = VELOCITY_TO_SOURCE_POSITION[velocity]
    lines = [
        "vars: x, y, z, t, u, w, a, b, c",
        "",
        f"# Isolated-hole one-step motion: {velocity}.",
        "# The support of the single dead cell is fully resolved over the affected 3x3 window.",
        "",
        "f(0, 0, 0, 0, 0, 0, 0, 0, 0) = 0;",
        "f(1, 1, 1, 1, 1, 1, 1, 1, 1) = 1;",
    ]
    for position in POSITION_ORDER:
        value = "0" if position == target_position else "1"
        lines.append(f"f({_single_zero_args(position)}) = {value};")
    return "\n".join(lines) + "\n"


def main() -> int:
    for velocity in VELOCITY_TO_SOURCE_POSITION:
        particle_path = IDENTITIES_DIR / f"isolated_particle_{velocity}.func"
        hole_path = IDENTITIES_DIR / f"isolated_hole_{velocity}.func"
        particle_path.write_text(_render_particle_spec(velocity))
        hole_path.write_text(_render_hole_spec(velocity))
        print(f"Wrote {particle_path}")
        print(f"Wrote {hole_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

