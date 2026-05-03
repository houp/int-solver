from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NUMBER_CONSERVING_EQUATIONS = ROOT / "equations" / "dedup" / "simplified_equations.txt"
CENTER_BLIND_EQUATIONS = ROOT / "equations" / "dedup" / "center_blind_equations.txt"
VARIABLE_ORDER = ("x", "y", "z", "t", "u", "w", "a", "b", "c")
SHIFT_POSITIONS = ("x", "y", "z", "t", "w", "a", "b", "c")


def parse_sparse_equations(path: Path) -> list[tuple[list[tuple[int, int]], int]]:
    equations: list[tuple[list[tuple[int, int]], int]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().replace(" ", "")
        if not line:
            continue
        lhs, rhs_text = line.split("=")
        rhs = int(rhs_text)
        terms: list[tuple[int, int]] = []
        index = 0
        while index < len(lhs):
            sign = 1
            while index < len(lhs) and lhs[index] in "+-":
                if lhs[index] == "-":
                    sign *= -1
                index += 1

            coeff = 1
            start = index
            while index < len(lhs) and lhs[index].isdigit():
                index += 1
            if index > start:
                coeff = int(lhs[start:index])
                if index < len(lhs) and lhs[index] == "*":
                    index += 1

            if lhs[index : index + 2] != "x_":
                raise ValueError(f"Malformed sparse term in {path}: {raw_line!r}")
            index += 2
            start = index
            while index < len(lhs) and lhs[index].isdigit():
                index += 1
            variable = int(lhs[start:index])
            terms.append((variable, sign * coeff))
        equations.append((terms, rhs))
    return equations


def rule_values_for_shift(position: str) -> dict[int, int]:
    bit_index = len(VARIABLE_ORDER) - 1 - VARIABLE_ORDER.index(position)
    return {entry: (entry >> bit_index) & 1 for entry in range(1, 511)}


def satisfies(equations: list[tuple[list[tuple[int, int]], int]], values: dict[int, int]) -> bool:
    for terms, rhs in equations:
        lhs = sum(coeff * values[var] for var, coeff in terms)
        if lhs != rhs:
            return False
    return True


class KnownRuleFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.number_conserving = parse_sparse_equations(NUMBER_CONSERVING_EQUATIONS)
        cls.center_blind = parse_sparse_equations(CENTER_BLIND_EQUATIONS)

    def test_all_neighbor_shifts_are_number_conserving_and_center_blind(self) -> None:
        for position in SHIFT_POSITIONS:
            values = rule_values_for_shift(position)
            self.assertTrue(
                satisfies(self.number_conserving, values),
                msg=f"shift-by-{position} should satisfy simplified_equations.txt",
            )
            self.assertTrue(
                satisfies(self.center_blind, values),
                msg=f"shift-by-{position} should satisfy center_blind_equations.txt",
            )


if __name__ == "__main__":
    unittest.main()
