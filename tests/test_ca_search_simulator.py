import importlib.util
import unittest

from ca_search.lut import identity_rule_bits, rigid_shift_rule_bits
from ca_search.simulator import NumpyBackend, collect_metric_series, create_backend


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "numpy is required for simulator tests")
class SimulatorTests(unittest.TestCase):
    def test_auto_backend_prefers_numpy(self):
        backend = create_backend("auto")
        self.assertEqual(backend.name, "numpy")

    def test_identity_rule_keeps_grid_unchanged(self):
        backend = NumpyBackend()
        states = backend.asarray([[[0, 1, 0], [1, 1, 0], [0, 0, 1]]], dtype="uint8")
        rules = backend.asarray([identity_rule_bits()], dtype="uint8")
        next_state = backend.to_numpy(backend.step_pairwise(states, rules))
        self.assertEqual(next_state.tolist(), [[[0, 1, 0], [1, 1, 0], [0, 0, 1]]])

    def test_north_shift_moves_particle_up(self):
        backend = NumpyBackend()
        states = backend.asarray([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype="uint8")
        rules = backend.asarray([rigid_shift_rule_bits("north")], dtype="uint8")
        next_state = backend.to_numpy(backend.step_pairwise(states, rules))
        self.assertEqual(next_state.tolist(), [[[0, 1, 0], [0, 0, 0], [0, 0, 0]]])

    def test_metric_series_for_identity_has_zero_activity(self):
        rules = [identity_rule_bits()]
        states = [[[0, 1], [1, 0]]]
        metrics = collect_metric_series("numpy", rules, states, steps=3)
        self.assertEqual(metrics.activity, [0.0, 0.0, 0.0])
        self.assertEqual(metrics.density, [0.5, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
