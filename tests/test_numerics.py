"""The original 0.1 numerics API (routerlab.numerics), kept for the learning material."""
import unittest

import numpy as np

from routerlab.numerics import projection_oracle, round_bf16, routing_diagnostics, topk_indices


class NumericalContractTests(unittest.TestCase):
    def test_bf16_rounds_halfway_to_even(self):
        # BF16 spacing above one is 1/128. Test both directions of ties-to-even.
        x = np.array([1 + 1/256, 1 + 3/256, -1 - 1/256], dtype=np.float32)
        np.testing.assert_array_equal(round_bf16(x), [1., 1 + 2/128, -1.])

    def test_bf16_special_values_and_no_mutation(self):
        x = np.array([0., -0., np.inf, -np.inf, np.nan], dtype=np.float32)
        original_bits = x.view(np.uint32).copy()
        result = round_bf16(x)
        self.assertTrue(np.signbit(result[1]))
        self.assertTrue(np.isposinf(result[2]))
        self.assertTrue(np.isneginf(result[3]))
        self.assertTrue(np.isnan(result[4]))
        np.testing.assert_array_equal(x.view(np.uint32), original_bits)

    def test_downcast_can_collapse_a_selection_boundary(self):
        scores = np.array([[1.0001, 1.0002, .2, -.1]], dtype=np.float32)
        self.assertEqual(topk_indices(scores, 1)[0, 0], 1)
        self.assertEqual(topk_indices(round_bf16(scores), 1)[0, 0], 0)
        metrics = routing_diagnostics(scores, round_bf16(scores), 1)
        self.assertEqual(metrics["near_boundary_rows"], 1)
        self.assertEqual(metrics["unexpected_changed_rows"], 0)

    def test_correction_bias_rounding_can_remove_a_distinction(self):
        raw = np.zeros((1, 4), dtype=np.float32)
        bias = np.array([1.0001, 1.0002, -.3, -.4], dtype=np.float32)
        self.assertNotEqual(int(topk_indices(raw + bias, 1)[0, 0]),
                            int(topk_indices(raw + round_bf16(bias), 1)[0, 0]))

    def test_projection_handles_noncontiguous_inputs(self):
        x = np.arange(30, dtype=np.float32).reshape(3, 10)[:, ::2]
        w = np.arange(40, dtype=np.float32).reshape(4, 10)[:, ::2]
        expected = np.array([[sum(float(a)*float(b) for a, b in zip(row, expert, strict=True))
                              for expert in w] for row in x])
        np.testing.assert_array_equal(projection_oracle(x, w), expected)

    def test_cancellation_oracle_preserves_small_term(self):
        x = np.array([[2**24, 1., -2**24]], dtype=np.float32)
        w = np.ones((2, 3), dtype=np.float32)
        np.testing.assert_array_equal(projection_oracle(x, w), [[1., 1.]])

    def test_margin_certificate_on_many_perturbations(self):
        rng = np.random.default_rng(42)
        reference = rng.normal(size=(200, 32))
        candidate = reference + rng.uniform(-.01, .01, reference.shape)
        metrics = routing_diagnostics(reference, candidate, 4)
        self.assertGreater(metrics["certified_rows"], 0)
        self.assertEqual(metrics["unexpected_changed_rows"], 0)

    def test_topk_measures_sets_not_order(self):
        a = np.array([[4., 3., 0., -1.]])
        b = np.array([[3., 4., 0., -1.]])
        self.assertEqual(routing_diagnostics(a, b, 2)["topk_set_match_fraction"], 1.)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            projection_oracle(np.ones((2, 3)), np.ones((4, 2)))
        with self.assertRaises(ValueError):
            routing_diagnostics([[1., np.nan]], [[1., 0.]], 1)
        with self.assertRaises(ValueError):
            topk_indices(np.ones((1, 3)), 4)  # k == E is valid (degenerate) since 0.2


if __name__ == "__main__":
    unittest.main()
