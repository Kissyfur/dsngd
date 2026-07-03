import unittest

import numpy as np

from src.families import CategoricalCoordinate, GaussianKnownVarianceCoordinate


class CategoricalCoordinateTests(unittest.TestCase):
    def test_natural_and_expectation_coordinates_round_trip(self):
        family = CategoricalCoordinate(4)
        expectation = np.array([0.2, 0.3, 0.1])

        natural = family.natural_from_expectation(expectation)
        recovered = family.expectation_from_natural(natural)

        np.testing.assert_allclose(recovered, expectation)

    def test_log_density_matches_probabilities(self):
        family = CategoricalCoordinate(3)
        expectation = np.array([0.2, 0.5])
        natural = family.natural_from_expectation(expectation)

        log_density = family.log_density(np.array([0, 1, 2]), natural)

        np.testing.assert_allclose(np.exp(log_density), np.array([0.2, 0.5, 0.3]))

    def test_dual_score_uses_canonical_baseline(self):
        family = CategoricalCoordinate(3)
        expectation = np.array([0.2, 0.5])

        score = family.dual_score(np.array([0, 1, 2]), expectation)

        expected = np.array(
            [
                [5.0, 0.0],
                [0.0, 2.0],
                [-1.0 / 0.3, -1.0 / 0.3],
            ]
        )
        np.testing.assert_allclose(score, expected)


class GaussianKnownVarianceCoordinateTests(unittest.TestCase):
    def test_natural_and_expectation_coordinates_round_trip(self):
        family = GaussianKnownVarianceCoordinate(variance=2.0)
        expectation = np.array([1.5])

        natural = family.natural_from_expectation(expectation)
        recovered = family.expectation_from_natural(natural)

        np.testing.assert_allclose(recovered, expectation)

    def test_log_density_matches_normal_density(self):
        family = GaussianKnownVarianceCoordinate(variance=4.0)
        expectation = np.array([1.0])
        natural = family.natural_from_expectation(expectation)

        log_density = family.log_density(np.array([-1.0, 1.0, 3.0]), natural)
        expected = -0.5 * np.log(2.0 * np.pi * 4.0) - 0.5 * np.array([4.0, 0.0, 4.0]) / 4.0

        np.testing.assert_allclose(log_density, expected)

    def test_dual_score_matches_finite_difference(self):
        family = GaussianKnownVarianceCoordinate(variance=2.0)
        x = 1.25
        expectation = np.array([0.4])
        epsilon = 1e-6

        def log_density_at(mean):
            return family.log_density(x, family.natural_from_expectation(np.array([mean])))

        numerical = (log_density_at(expectation[0] + epsilon) - log_density_at(expectation[0] - epsilon)) / (
            2.0 * epsilon
        )

        np.testing.assert_allclose(family.dual_score(x, expectation), np.array([numerical]), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
