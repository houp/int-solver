import unittest
from pathlib import Path

from ca_search.catalog import PropertySpec, RuleCatalog, RuleRecord
from ca_search.lut import identity_rule_bits, lut_bits_to_hex, rigid_shift_rule_bits
from ca_search.simple_filters import summarize_catalog_rule, summarize_simple_rule


class SearchFilterTests(unittest.TestCase):
    def test_identity_rule_summary(self):
        summary = summarize_simple_rule(identity_rule_bits(), [])
        self.assertEqual(summary.rigid_source_position, "u")
        self.assertEqual(summary.rigid_velocity, "static")
        self.assertEqual(summary.isolated_particle_velocity, "static")
        self.assertEqual(summary.isolated_hole_velocity, "static")
        self.assertIn("identity", summary.tags)

    def test_north_shift_summary(self):
        summary = summarize_simple_rule(rigid_shift_rule_bits("north"), [])
        self.assertEqual(summary.rigid_source_position, "b")
        self.assertEqual(summary.rigid_velocity, "north")
        self.assertEqual(summary.isolated_particle_velocity, "north")
        self.assertEqual(summary.isolated_hole_velocity, "north")
        self.assertIn("rigid_shift:north", summary.tags)

    def test_catalog_summary_uses_property_masks(self):
        catalog = RuleCatalog(
            properties=(PropertySpec(bit=0, name="von_neumann", path="von_neumann_equations.txt"),),
            rules=(
                RuleRecord(
                    id=0,
                    legacy_index=0,
                    stable_index=0,
                    stable_id="deadbeef" * 8,
                    stable_id_short="deadbeefdead",
                    mask=1,
                    lut_hex=lut_bits_to_hex(identity_rule_bits()),
                    ones=sum(identity_rule_bits()),
                ),
            ),
            source_path=Path("dataset.json"),
        )
        summary = summarize_catalog_rule(catalog, catalog.rules[0])
        self.assertIn("identity", summary.tags)
        self.assertNotIn("embedded_von_neumann_nonshift", summary.tags)


if __name__ == "__main__":
    unittest.main()
