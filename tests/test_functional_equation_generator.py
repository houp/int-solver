from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from generate_functional_equation_system import instantiate_all_identities, load_spec


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "identities" / "number_conserving.func"
IDENTITY_DIR = ROOT / "identities"
BUILTIN_SPECS = sorted(path for path in IDENTITY_DIR.glob("*.func"))


class FunctionalEquationGeneratorTests(unittest.TestCase):
    def test_default_spec_has_expected_shape(self) -> None:
        spec = load_spec(DEFAULT_SPEC)
        self.assertEqual(spec.variable_order, ("x", "y", "z", "t", "u", "w", "a", "b", "c"))
        self.assertEqual(len(spec.identities), 3)
        self.assertEqual(spec.identities[0].free_variables, spec.variable_order)
        self.assertEqual(spec.identities[1].free_variables, ())
        self.assertEqual(spec.identities[2].free_variables, ())

    def test_all_builtin_family_specs_load(self) -> None:
        self.assertGreaterEqual(len(BUILTIN_SPECS), 10)
        self.assertIn(DEFAULT_SPEC, BUILTIN_SPECS)

        for spec_path in BUILTIN_SPECS:
            spec = load_spec(spec_path)
            self.assertEqual(
                spec.variable_order,
                ("x", "y", "z", "t", "u", "w", "a", "b", "c"),
                msg=f"bad variable order in {spec_path.name}",
            )
            self.assertGreaterEqual(len(spec.identities), 3, msg=f"too few identities in {spec_path.name}")
            self.assertEqual(spec.identities[-2].free_variables, (), msg=f"missing zero boundary in {spec_path.name}")
            self.assertEqual(spec.identities[-1].free_variables, (), msg=f"missing one boundary in {spec_path.name}")

    def test_default_spec_reproduces_checked_in_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sparse_out = tmpdir_path / "simplified_equations.txt"
            matrix_out = tmpdir_path / "simplified_equations_matrix.csv"
            raw_out = tmpdir_path / "all_512_raw_equations.txt"

            subprocess.run(
                [
                    sys.executable,
                    "src/generate_functional_equation_system.py",
                    str(DEFAULT_SPEC),
                    "--skip-rank",
                    "--sparse-out",
                    str(sparse_out),
                    "--matrix-out",
                    str(matrix_out),
                    "--raw-out",
                    str(raw_out),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                sparse_out.read_text(encoding="utf-8"),
                (ROOT / "equations" / "dedup" / "simplified_equations.txt").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                matrix_out.read_text(encoding="utf-8"),
                (ROOT / "equations" / "matrices" / "simplified_equations_matrix.csv").read_text(encoding="utf-8"),
            )

    def test_diagonal_von_neumann_alias_matches_diagonal_only(self) -> None:
        diagonal_only = load_spec(IDENTITY_DIR / "diagonal_only.func")
        diagonal_vn = load_spec(IDENTITY_DIR / "diagonal_von_neumann.func")

        self.assertEqual(diagonal_only.variable_order, diagonal_vn.variable_order)

        equations_only = instantiate_all_identities(diagonal_only)
        equations_vn = instantiate_all_identities(diagonal_vn)
        self.assertEqual(equations_only, equations_vn)

    def test_monotone_spec_emits_implications(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sparse_out = tmpdir_path / "monotone_equations.txt"

            subprocess.run(
                [
                    sys.executable,
                    "src/generate_functional_equation_system.py",
                    str(IDENTITY_DIR / "monotone.func"),
                    "--skip-rank",
                    "--no-substitute-fixed",
                    "--sparse-out",
                    str(sparse_out),
                    "--matrix-out",
                    str(tmpdir_path / "monotone_matrix.csv"),
                    "--raw-out",
                    str(tmpdir_path / "monotone_raw.txt"),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            sparse_lines = sparse_out.read_text(encoding="utf-8").splitlines()
            self.assertTrue(any("<=" in line for line in sparse_lines))
            self.assertIn("x_511 = 1", sparse_out.read_text(encoding="utf-8"))

    def test_center_permutive_spec_emits_disequality_as_affine_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sparse_out = tmpdir_path / "center_permutive_equations.txt"

            subprocess.run(
                [
                    sys.executable,
                    "src/generate_functional_equation_system.py",
                    str(IDENTITY_DIR / "center_permutive.func"),
                    "--skip-rank",
                    "--sparse-out",
                    str(sparse_out),
                    "--matrix-out",
                    str(tmpdir_path / "center_permutive_matrix.csv"),
                    "--raw-out",
                    str(tmpdir_path / "center_permutive_raw.txt"),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            sparse_lines = sparse_out.read_text(encoding="utf-8").splitlines()
            self.assertTrue(any(" + x_" in line and " = 1" in line for line in sparse_lines))


if __name__ == "__main__":
    unittest.main()
