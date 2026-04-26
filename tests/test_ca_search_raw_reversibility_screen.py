import tempfile
import unittest
from pathlib import Path

import numpy as np

from ca_search.lut import (
    embedded_von_neumann_traffic_rule_bits,
    identity_rule_bits,
    rigid_shift_rule_bits,
    rule_stable_id,
)
from ca_search.raw_reversibility_screen import screen_raw_catalog_for_reversibility


def _pack_raw_rule(bits: list[int]) -> bytes:
    payload = np.asarray(bits[1:511], dtype=np.uint8)
    packed = np.packbits(payload, bitorder="little")
    if packed.size < 64:
        packed = np.concatenate((packed, np.zeros(64 - packed.size, dtype=np.uint8)))
    return packed[:64].tobytes()


class RawReversibilityScreenTests(unittest.TestCase):
    def test_raw_screen_matches_expected_small_catalog(self):
        rules = [
            identity_rule_bits(),
            rigid_shift_rule_bits("north"),
            embedded_von_neumann_traffic_rule_bits("east"),
        ]
        expected_survivors = sorted(rule_stable_id(rule) for rule in rules[:2])

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "sample.bin"
            with raw_path.open("wb") as handle:
                for rule in rules:
                    handle.write(_pack_raw_rule(rule))

            summary = screen_raw_catalog_for_reversibility(raw_path, records_per_chunk=2, stage_populations=(2, 3, 4))

        self.assertEqual(summary.total_rules, 3)
        self.assertEqual(summary.surviving_rule_count, 2)
        self.assertEqual([item.stable_id for item in summary.survivors], expected_survivors)
        self.assertEqual(summary.stages[0].stage, "particle-sector:2@4x4")
        self.assertEqual(summary.stages[0].tested_rules, 3)
        self.assertEqual(summary.stages[0].rejected_rules, 1)
        self.assertEqual(summary.stages[1].tested_rules, 2)
        self.assertEqual(summary.stages[2].tested_rules, 2)


if __name__ == "__main__":
    unittest.main()
