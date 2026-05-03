import unittest
from pathlib import Path

import numpy as np

from ca_search.binary_catalog import BinaryCatalog
from ca_search.lut import (
    embedded_von_neumann_traffic_rule_bits,
    identity_rule_bits,
    rigid_shift_rule_bits,
    rule_stable_id,
)
from ca_search.reversibility import run_exact_reversibility_screen
from ca_search.reversibility_catalog import screen_catalog_exact_reversibility


def _make_catalog(rules: list[list[int]]) -> BinaryCatalog:
    lut_bits = np.asarray(rules, dtype=np.uint8)
    total = lut_bits.shape[0]
    return BinaryCatalog(
        properties=(),
        masks=np.zeros(total, dtype=np.uint64),
        lut_bits=lut_bits,
        ids=np.arange(total, dtype=np.int64),
        stable_indices=np.arange(total, dtype=np.int64),
        stable_ids=tuple(rule_stable_id(rule) for rule in rules),
        stable_order=np.arange(total, dtype=np.int64),
        metadata_path=Path("synthetic.json"),
        binary_path=Path("synthetic.bin"),
    )


class ReversibilityCatalogTests(unittest.TestCase):
    def test_catalog_screen_rejects_known_nonreversible_rule(self):
        rules = [
            identity_rule_bits(),
            rigid_shift_rule_bits("north"),
            embedded_von_neumann_traffic_rule_bits("east"),
        ]
        catalog = _make_catalog(rules)

        result = screen_catalog_exact_reversibility(
            catalog,
            sector_grid=(4, 4),
            particle_populations=(2,),
            hole_populations=(),
            torus_grids=((3, 3),),
            rule_batch_size=2,
        )

        self.assertEqual(result.total_rules, 3)
        self.assertEqual(result.trivial_reversible_rules, 2)
        self.assertEqual(result.nontrivial_rules, 1)
        self.assertEqual(result.surviving_rule_count, 0)
        self.assertEqual(result.surviving_stable_ids, [])
        self.assertEqual(len(result.stages), 2)
        self.assertEqual(result.stages[0].stage, "particle-sector:2@4x4")
        self.assertEqual(result.stages[0].tested_rules, 1)
        self.assertEqual(result.stages[0].rejected_rules, 1)
        self.assertEqual(result.stages[0].surviving_rules, 0)
        self.assertEqual(result.stages[1].stage, "torus:3x3")
        self.assertEqual(result.stages[1].tested_rules, 0)

    def test_catalog_screen_matches_single_rule_screen_for_nontrivial_rule(self):
        traffic_rule = embedded_von_neumann_traffic_rule_bits("east")
        catalog = _make_catalog([traffic_rule])

        batch_result = screen_catalog_exact_reversibility(
            catalog,
            sector_grid=(4, 4),
            particle_populations=(2,),
            hole_populations=(1,),
            torus_grids=((3, 3),),
            rule_batch_size=1,
        )
        single_result = run_exact_reversibility_screen(
            traffic_rule,
            sector_grid=(4, 4),
            particle_populations=(2,),
            hole_populations=(1,),
            torus_grids=((3, 3),),
            sector_batch_size=128,
            torus_batch_size=512,
        )

        self.assertEqual(single_result.rejection_stage, "particle-sector:2")
        self.assertEqual(batch_result.surviving_rule_count, 0)
        self.assertEqual(batch_result.stages[0].stage, "particle-sector:2@4x4")
        self.assertEqual(batch_result.stages[0].rejected_rules, 1)


if __name__ == "__main__":
    unittest.main()
