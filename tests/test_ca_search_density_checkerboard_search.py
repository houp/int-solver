import importlib.util
import unittest

from ca_search.density_checkerboard_search import (
    build_checkerboard_prephase_schedules,
    default_checkerboard_block_library,
    rank_checkerboard_prephase_schedules,
)
from ca_search.density_classification import moore_majority_rule_bits
from ca_search.lut import rigid_rule_bits


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "numpy is required for checkerboard search tests")
class DensityCheckerboardSearchTests(unittest.TestCase):
    def test_build_checkerboard_prephase_schedules(self):
        schedules = build_checkerboard_prephase_schedules(
            blocks=(("shift_t",), ("shift_t", "shift_w")),
            steps_per_rule_values=(1, 2),
            block_repetitions=(2,),
            amplifier_name="majority_moore",
            amplifier_steps_values=(3, 4),
        )
        self.assertEqual(len(schedules), 8)

    def test_rank_checkerboard_prephase_schedules(self):
        schedules = build_checkerboard_prephase_schedules(
            blocks=(("shift_t",),),
            steps_per_rule_values=(1,),
            block_repetitions=(2,),
            amplifier_name="majority_moore",
            amplifier_steps_values=(1,),
        )
        rule_bank = {
            "shift_t": rigid_rule_bits("t"),
            "majority_moore": moore_majority_rule_bits(),
        }
        ranked = rank_checkerboard_prephase_schedules(
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
        self.assertEqual(len(ranked), 1)
        self.assertGreaterEqual(ranked[0].min_checkerboard_2x2_fraction, 0.0)
        self.assertLessEqual(ranked[0].min_checkerboard_2x2_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
