import unittest

from ca_search.lut import (
    embedded_von_neumann_traffic_rule_bits,
    identity_rule_bits,
    rigid_shift_rule_bits,
)
from ca_search.reversibility import (
    run_exact_reversibility_screen,
    test_sector_injective as sector_injective,  # alias avoids pytest collection
    test_small_torus_bijective as small_torus_bijective,
)


class ReversibilityTests(unittest.TestCase):
    def test_identity_is_injective_on_small_sector_and_torus(self):
        sector = sector_injective(identity_rule_bits(), width=4, height=4, population=2, batch_size=64)
        torus = small_torus_bijective(identity_rule_bits(), width=4, height=4, batch_size=256)
        self.assertTrue(sector.is_injective)
        self.assertIsNone(sector.witness)
        self.assertTrue(torus.is_bijective)
        self.assertIsNone(torus.witness)

    def test_shift_is_treated_as_trivial_reversible(self):
        report = run_exact_reversibility_screen(rigid_shift_rule_bits("north"))
        self.assertTrue(report.trivial_reversible)
        self.assertIsNone(report.rejection_stage)
        self.assertEqual(report.sector_results, [])
        self.assertEqual(report.torus_results, [])

    def test_embedded_traffic_rule_fails_sector_injectivity(self):
        result = sector_injective(
            embedded_von_neumann_traffic_rule_bits("east"),
            width=4,
            height=4,
            population=2,
            batch_size=128,
        )
        self.assertFalse(result.is_injective)
        self.assertIsNotNone(result.witness)

    def test_embedded_traffic_rule_fails_small_torus_bijectivity(self):
        result = small_torus_bijective(
            embedded_von_neumann_traffic_rule_bits("east"),
            width=4,
            height=4,
            batch_size=512,
        )
        self.assertFalse(result.is_bijective)
        self.assertIsNotNone(result.witness)

    def test_screen_reports_rejection_stage_for_nonreversible_traffic_rule(self):
        report = run_exact_reversibility_screen(
            embedded_von_neumann_traffic_rule_bits("east"),
            sector_grid=(4, 4),
            particle_populations=(2,),
            hole_populations=(),
            torus_grids=(),
            sector_batch_size=128,
        )
        self.assertFalse(report.trivial_reversible)
        self.assertEqual(report.rejection_stage, "particle-sector:2")
        self.assertEqual(len(report.sector_results), 1)
        self.assertFalse(report.sector_results[0].is_injective)


if __name__ == "__main__":
    unittest.main()
