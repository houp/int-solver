import importlib.util
import unittest
from pathlib import Path

import numpy as np

from ca_search.binary_catalog import BinaryCatalog, PropertyInfo
from ca_search.density_classification import (
    ScheduleSpec,
    embedded_diagonal_traffic_rule_bits,
    evaluate_schedule,
    moore_threshold_rule_bits,
    moore_majority_rule_bits,
    screen_finite_switch_preprocessors,
    screen_repeated_block_preprocessors,
    screen_preprocessors_balanced,
    screen_preprocessors,
    select_catalog_indices_by_property_names,
    von_neumann_threshold_rule_bits,
    von_neumann_majority_rule_bits,
)
from ca_search.lut import neighborhood_index, rigid_rule_bits, rule_stable_id


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "numpy is required for density-classification tests")
class DensityClassificationTests(unittest.TestCase):
    def test_moore_majority_rule_bits(self):
        bits = moore_majority_rule_bits()
        self.assertEqual(bits[neighborhood_index(0, 0, 0, 0, 0, 0, 0, 0, 0)], 0)
        self.assertEqual(bits[neighborhood_index(1, 1, 1, 1, 1, 0, 0, 0, 0)], 1)
        self.assertEqual(bits[neighborhood_index(1, 1, 1, 1, 0, 0, 0, 0, 0)], 0)

    def test_von_neumann_majority_ignores_corners(self):
        bits = von_neumann_majority_rule_bits()
        center_cross = neighborhood_index(0, 1, 0, 1, 1, 0, 0, 0, 0)
        only_corners = neighborhood_index(1, 0, 1, 0, 0, 0, 1, 0, 1)
        self.assertEqual(bits[center_cross], 1)
        self.assertEqual(bits[only_corners], 0)

    def test_threshold_rule_bits(self):
        moore = moore_threshold_rule_bits(6)
        self.assertEqual(moore[neighborhood_index(1, 1, 1, 1, 1, 0, 0, 0, 0)], 0)
        self.assertEqual(moore[neighborhood_index(1, 1, 1, 1, 1, 1, 0, 0, 0)], 1)

        vn = von_neumann_threshold_rule_bits(4)
        self.assertEqual(vn[neighborhood_index(0, 1, 0, 1, 1, 0, 0, 0, 0)], 0)
        self.assertEqual(vn[neighborhood_index(0, 1, 0, 1, 1, 1, 0, 1, 0)], 1)

    def test_diagonal_traffic_moves_particle_northeast(self):
        rule = embedded_diagonal_traffic_rule_bits("northeast")
        # Only southwest source occupied, center empty -> center becomes 1.
        idx = neighborhood_index(0, 0, 0, 0, 0, 0, 1, 0, 0)
        self.assertEqual(rule[idx], 1)
        # Center occupied, northeast empty -> particle moves out, center becomes 0.
        idx = neighborhood_index(0, 0, 0, 0, 1, 0, 0, 0, 0)
        self.assertEqual(rule[idx], 0)

    def test_majority_only_schedule_classifies_uniform_batches(self):
        initial_states = np.asarray(
            [
                np.zeros((4, 4), dtype=np.uint8),
                np.ones((4, 4), dtype=np.uint8),
            ]
        )
        schedule = ScheduleSpec("majority_only", (("majority_moore", 3),))
        report = evaluate_schedule(
            "numpy",
            schedule,
            {"majority_moore": moore_majority_rule_bits()},
            initial_states,
            majority_metric="moore",
        )
        self.assertEqual(report.tied_initial_states, 0)
        self.assertEqual(report.final_majority_accuracy, 1.0)
        self.assertEqual(report.final_consensus_accuracy, 1.0)
        self.assertEqual(report.final_consensus_rate, 1.0)

    def test_checkerboard_metrics_detect_order(self):
        checker = (np.indices((6, 6)).sum(axis=0) & 1).astype(np.uint8)
        initial_states = np.asarray([checker, 1 - checker], dtype=np.uint8)
        schedule = ScheduleSpec("shift_then_majority", (("shift_t", 1), ("majority_moore", 1)))
        report = evaluate_schedule(
            "numpy",
            schedule,
            {"shift_t": rigid_rule_bits("t"), "majority_moore": moore_majority_rule_bits()},
            initial_states,
            majority_metric="moore",
        )
        self.assertGreaterEqual(report.preprocess_checkerboard_alignment, 0.9)
        self.assertGreaterEqual(report.preprocess_orthogonal_disagreement, 0.9)
        self.assertGreaterEqual(report.preprocess_checkerboard_2x2_fraction, 0.9)

    def test_preprocessor_screen_excludes_trivial_identity(self):
        identity = np.asarray(rigid_rule_bits("u"), dtype=np.uint8)
        candidate = np.asarray(rigid_rule_bits("t"), dtype=np.uint8)
        catalog = BinaryCatalog(
            properties=(PropertyInfo(bit=0, name="dummy", path="dummy.txt"),),
            masks=np.asarray([0, 0], dtype=np.uint64),
            lut_bits=np.stack([identity, candidate], axis=0),
            ids=np.asarray([0, 1], dtype=np.int64),
            stable_indices=np.asarray([0, 1], dtype=np.int64),
            stable_ids=(rule_stable_id(identity.tolist()), rule_stable_id(candidate.tolist())),
            stable_order=np.asarray([0, 1], dtype=np.int64),
            metadata_path=Path("meta.json"),
            binary_path=Path("rules.bin"),
        )
        results = screen_preprocessors(
            catalog,
            backend_name="numpy",
            width=8,
            height=8,
            probability=0.49,
            trials=4,
            seed=0,
            preprocess_steps=1,
            max_rules=None,
            legacy_indices=[0, 1],
            require_nontrivial=True,
            rule_batch_size=2,
            majority_metric="moore",
            top_k=10,
        )
        self.assertEqual(results, [])

    def test_select_catalog_indices_by_property_names_matches_union(self):
        identity = np.asarray(rigid_rule_bits("u"), dtype=np.uint8)
        east = np.asarray(rigid_rule_bits("t"), dtype=np.uint8)
        west = np.asarray(rigid_rule_bits("w"), dtype=np.uint8)
        catalog = BinaryCatalog(
            properties=(
                PropertyInfo(bit=0, name="alpha", path="alpha.txt"),
                PropertyInfo(bit=1, name="beta", path="beta.txt"),
            ),
            masks=np.asarray([0b01, 0b10, 0b11], dtype=np.uint64),
            lut_bits=np.stack([identity, east, west], axis=0),
            ids=np.asarray([0, 1, 2], dtype=np.int64),
            stable_indices=np.asarray([0, 1, 2], dtype=np.int64),
            stable_ids=(
                rule_stable_id(identity.tolist()),
                rule_stable_id(east.tolist()),
                rule_stable_id(west.tolist()),
            ),
            stable_order=np.asarray([0, 1, 2], dtype=np.int64),
            metadata_path=Path("meta.json"),
            binary_path=Path("rules.bin"),
        )
        self.assertEqual(select_catalog_indices_by_property_names(catalog, ["alpha"]), [0, 2])
        self.assertEqual(select_catalog_indices_by_property_names(catalog, ["alpha", "beta"]), [0, 1, 2])

    def test_balanced_preprocessor_screen_reports_paired_metrics(self):
        east = np.asarray(rigid_rule_bits("t"), dtype=np.uint8)
        west = np.asarray(rigid_rule_bits("w"), dtype=np.uint8)
        catalog = BinaryCatalog(
            properties=(PropertyInfo(bit=0, name="dummy", path="dummy.txt"),),
            masks=np.asarray([1, 1], dtype=np.uint64),
            lut_bits=np.stack([east, west], axis=0),
            ids=np.asarray([0, 1], dtype=np.int64),
            stable_indices=np.asarray([0, 1], dtype=np.int64),
            stable_ids=(rule_stable_id(east.tolist()), rule_stable_id(west.tolist())),
            stable_order=np.asarray([0, 1], dtype=np.int64),
            metadata_path=Path("meta.json"),
            binary_path=Path("rules.bin"),
        )
        results = screen_preprocessors_balanced(
            catalog,
            backend_name="numpy",
            width=6,
            height=6,
            probabilities=(0.49, 0.51),
            trials=4,
            seed=0,
            preprocess_steps=1,
            majority_tail=1,
            include_property_names=["dummy"],
            require_nontrivial=False,
            rule_batch_size=2,
            top_k=2,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].probabilities, (0.49, 0.51))
        self.assertEqual(len(results[0].per_probability), 2)
        self.assertGreaterEqual(results[0].min_final_majority_accuracy, 0.0)
        self.assertLessEqual(results[0].final_majority_accuracy_gap, 1.0)

    def test_repeated_block_screen_reports_consensus_metrics(self):
        east = np.asarray(rigid_rule_bits("t"), dtype=np.uint8)
        west = np.asarray(rigid_rule_bits("w"), dtype=np.uint8)
        catalog = BinaryCatalog(
            properties=(PropertyInfo(bit=0, name="dummy", path="dummy.txt"),),
            masks=np.asarray([1, 1], dtype=np.uint64),
            lut_bits=np.stack([east, west], axis=0),
            ids=np.asarray([0, 1], dtype=np.int64),
            stable_indices=np.asarray([0, 1], dtype=np.int64),
            stable_ids=(rule_stable_id(east.tolist()), rule_stable_id(west.tolist())),
            stable_order=np.asarray([0, 1], dtype=np.int64),
            metadata_path=Path("meta.json"),
            binary_path=Path("rules.bin"),
        )
        results = screen_repeated_block_preprocessors(
            catalog,
            backend_name="numpy",
            width=6,
            height=6,
            probabilities=(0.49, 0.51),
            trials=4,
            seed=0,
            preprocess_steps=1,
            amplifier_steps=1,
            repetitions=2,
            amplifier_name="majority_moore",
            amplifier_bits=moore_majority_rule_bits(),
            include_property_names=["dummy"],
            require_nontrivial=False,
            rule_batch_size=2,
            top_k=2,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].amplifier_name, "majority_moore")
        self.assertGreaterEqual(results[0].min_final_consensus_rate, 0.0)
        self.assertLessEqual(results[0].min_final_consensus_rate, 1.0)

    def test_finite_switch_screen_reports_switch_metrics(self):
        east = np.asarray(rigid_rule_bits("t"), dtype=np.uint8)
        west = np.asarray(rigid_rule_bits("w"), dtype=np.uint8)
        catalog = BinaryCatalog(
            properties=(PropertyInfo(bit=0, name="dummy", path="dummy.txt"),),
            masks=np.asarray([1, 1], dtype=np.uint64),
            lut_bits=np.stack([east, west], axis=0),
            ids=np.asarray([0, 1], dtype=np.int64),
            stable_indices=np.asarray([0, 1], dtype=np.int64),
            stable_ids=(rule_stable_id(east.tolist()), rule_stable_id(west.tolist())),
            stable_order=np.asarray([0, 1], dtype=np.int64),
            metadata_path=Path("meta.json"),
            binary_path=Path("rules.bin"),
        )
        results = screen_finite_switch_preprocessors(
            catalog,
            backend_name="numpy",
            width=6,
            height=6,
            probabilities=(0.49, 0.51),
            trials=4,
            seed=0,
            preprocess_steps_options=(1, 2),
            amplifier_steps_options=(1, 2),
            amplifier_name="majority_vn",
            amplifier_bits=von_neumann_majority_rule_bits(),
            include_property_names=["dummy"],
            require_nontrivial=False,
            rule_batch_size=2,
            top_k=8,
        )
        self.assertEqual(len(results), 8)
        self.assertEqual(results[0].amplifier_name, "majority_vn")
        self.assertIn(results[0].preprocess_steps, (1, 2))
        self.assertIn(results[0].amplifier_steps, (1, 2))
        self.assertGreaterEqual(results[0].min_final_consensus_accuracy, 0.0)
        self.assertLessEqual(results[0].min_final_consensus_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
