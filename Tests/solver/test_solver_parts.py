import unittest

import numpy as np

from bl_ext.user_default.freeclipmocap.solver import gap_filling, one_euro_filter, rotations


class RotationTests(unittest.TestCase):
    def test_quaternion_matrix_round_trip(self) -> None:
        quaternions = rotations.normalized(np.random.default_rng(1).normal(size=(200, 4)))
        quaternions *= np.where(quaternions[:, :1] < 0, -1.0, 1.0)
        round_trip = rotations.quaternions_from_matrices(rotations.matrices_from_quaternions(quaternions))
        np.testing.assert_allclose(round_trip, quaternions, atol=1e-9)

    def test_rotation_between_maps_one_vector_onto_the_other(self) -> None:
        generator = np.random.default_rng(2)
        start = generator.normal(size=(100, 3))
        end = generator.normal(size=(100, 3))
        end[0] = -start[0]
        mapped = np.einsum("fij,fj->fi", rotations.rotation_between(start, end), rotations.normalized(start))
        np.testing.assert_allclose(mapped, rotations.normalized(end), atol=1e-9)

    def test_slerp_ends_and_middle(self) -> None:
        start = np.array([1.0, 0.0, 0.0, 0.0])
        end = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        np.testing.assert_allclose(rotations.slerp(start, end, 0.0), start, atol=1e-12)
        np.testing.assert_allclose(rotations.slerp(start, end, 1.0), end, atol=1e-12)
        middle = rotations.slerp(start, end, 0.5)
        np.testing.assert_allclose(middle, [np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)], atol=1e-12)

    def test_make_continuous_removes_sign_flips(self) -> None:
        quaternions = np.array([[1.0, 0.0, 0.0, 0.0], [-0.99, -0.1, 0.0, 0.0], [0.98, 0.2, 0.0, 0.0]])
        continuous = rotations.make_continuous(quaternions)
        self.assertTrue(np.all(np.sum(continuous[1:] * continuous[:-1], axis=1) > 0))


class GapFillingTests(unittest.TestCase):
    def test_hold_last_good_and_fill_the_start_with_the_first_good(self) -> None:
        values = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        found = np.array([False, True, False, False, True, False])
        np.testing.assert_array_equal(gap_filling.hold_last_good(values, found), [11.0, 11.0, 11.0, 11.0, 14.0, 14.0])

    def test_nothing_found(self) -> None:
        self.assertIsNone(gap_filling.hold_last_good(np.zeros(3), np.zeros(3, dtype=bool)))


class GapBlendingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.times = np.arange(100) / 30.0
        self.values = np.zeros((100, 1, 3))
        self.values[50:] = 1.0

    def blended(self, found: np.ndarray) -> np.ndarray:
        filled = gap_filling.blend_gaps(self.values, found, self.times, 1.0, 0.25)
        assert filled is not None
        return filled[:, 0, 0]

    def test_short_gap_eases_from_the_last_good_value_to_the_next(self) -> None:
        found = np.ones(100, dtype=bool)
        found[40:50] = False
        filled = self.blended(found)
        self.assertEqual(filled[39], 0.0)
        fraction = 6 / 11
        self.assertAlmostEqual(filled[45], fraction * fraction * (3.0 - 2.0 * fraction), places=6)
        self.assertTrue(np.all(np.diff(filled[39:51]) > 0.0))

    def test_long_gap_holds_then_eases_in_the_last_quarter_second(self) -> None:
        found = np.ones(100, dtype=bool)
        found[5:50] = False
        filled = self.blended(found)
        np.testing.assert_array_equal(filled[5:42], 0.0)
        self.assertTrue(np.all(np.diff(filled[42:51]) > 0.0))
        self.assertEqual(filled[50], 1.0)

    def test_gap_at_the_start_holds_the_first_good_value(self) -> None:
        found = np.zeros(100, dtype=bool)
        found[60:] = True
        np.testing.assert_array_equal(self.blended(found)[:60], 1.0)
        self.assertIsNone(gap_filling.blend_gaps(self.values, np.zeros(100, dtype=bool), self.times, 1.0, 0.25))


class OneEuroFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.timestamps = np.arange(300) / 30.0

    def test_still_points_stay_put(self) -> None:
        values = np.ones((300, 5, 3))
        np.testing.assert_allclose(one_euro_filter.smooth(values, self.timestamps, 0.5), values)

    def test_jitter_is_reduced_more_with_more_smoothing(self) -> None:
        jitter = np.random.default_rng(3).normal(scale=0.005, size=(300, 5, 3))
        light = np.std(one_euro_filter.smooth(jitter, self.timestamps, 0.2)[30:])
        strong = np.std(one_euro_filter.smooth(jitter, self.timestamps, 0.9)[30:])
        self.assertLess(strong, light)
        self.assertLess(light, np.std(jitter))

    def test_zero_smoothing_changes_nothing(self) -> None:
        values = np.random.default_rng(4).normal(size=(50, 2, 3))
        np.testing.assert_array_equal(one_euro_filter.smooth(values, self.timestamps[:50], 0.0), values)


if __name__ == "__main__":
    unittest.main()
