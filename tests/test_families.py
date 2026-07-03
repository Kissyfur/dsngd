import unittest

import numpy as np

from src.families import CategoricalCoordinate


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


if __name__ == "__main__":
    unittest.main()
