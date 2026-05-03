import importlib.util
import unittest

import numpy as np

from ca_search.density_classification import moore_majority_rule_bits
from ca_search.density_schedule_search import (
    make_prephase_repeated_block_schedules,
    make_preprocessor_chain_schedules,
    make_repeated_block_schedules,
    make_single_preprocess_schedules,
    rank_density_schedules,
)
from ca_search.lut import rigid_rule_bits


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "numpy is required for density schedule search tests")
class DensityScheduleSearchTests(unittest.TestCase):
    def test_schedule_generators(self):
        single = make_single_preprocess_schedules(
            ["shift_t"], ["majority_moore"], preprocess_steps=4, amplifier_steps=6
        )
        self.assertEqual(len(single), 2)

        chains = make_preprocessor_chain_schedules(
            ["shift_t", "shift_w"], ["majority_moore"], preprocess_steps=3, amplifier_steps=5
        )
        self.assertEqual(len(chains), 2)

        repeated = make_repeated_block_schedules(
            ["shift_t"], ["majority_moore"], preprocess_steps=2, amplifier_steps=3, repetitions=[2, 4]
        )
        self.assertEqual(len(repeated), 2)

        prephase = make_prephase_repeated_block_schedules(
            [("shift_t", "shift_w"), ("shift_t",)],
            ["majority_moore"],
            steps_per_rule=1,
            block_repetitions=[2],
            amplifier_steps=3,
        )
        self.assertEqual(len(prephase), 2)

    def test_rank_density_schedules(self):
        schedules = make_single_preprocess_schedules(
            ["shift_t"], ["majority_moore"], preprocess_steps=1, amplifier_steps=1
        )
        rule_bank = {
            "shift_t": rigid_rule_bits("t"),
            "majority_moore": moore_majority_rule_bits(),
        }
        ranked = rank_density_schedules(
            "numpy",
            schedules,
            rule_bank,
            width=6,
            height=6,
            probabilities=(0.49, 0.51),
            trials_per_probability=4,
            seed=0,
            top_k=2,
        )
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].probabilities, (0.49, 0.51))
        self.assertGreaterEqual(ranked[0].min_final_majority_accuracy, 0.0)
        self.assertLessEqual(ranked[0].final_majority_accuracy_gap, 1.0)


if __name__ == "__main__":
    unittest.main()
