import unittest

import numpy as np

from src.families import (
    CategoricalCoordinate,
    ExponentialMeanCoordinate,
    GaussianKnownVarianceCoordinate,
    GaussianUnknownVarianceCoordinate,
    MultivariateGaussianCoordinate,
    PoissonCoordinate,
)


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


class GaussianUnknownVarianceCoordinateTests(unittest.TestCase):
    def test_natural_and_expectation_coordinates_round_trip(self):
        family = GaussianUnknownVarianceCoordinate()
        expectation = np.array([1.5, 3.25])

        natural = family.natural_from_expectation(expectation)
        recovered = family.expectation_from_natural(natural)

        np.testing.assert_allclose(recovered, expectation)

    def test_log_density_matches_normal_density(self):
        family = GaussianUnknownVarianceCoordinate()
        mean = 1.0
        variance = 4.0
        expectation = np.array([mean, mean * mean + variance])
        natural = family.natural_from_expectation(expectation)

        log_density = family.log_density(np.array([-1.0, 1.0, 3.0]), natural)
        expected = -0.5 * np.log(2.0 * np.pi * variance) - 0.5 * np.array([4.0, 0.0, 4.0]) / variance

        np.testing.assert_allclose(log_density, expected)

    def test_dual_score_matches_finite_difference(self):
        family = GaussianUnknownVarianceCoordinate()
        x = 1.25
        expectation = np.array([0.4, 2.0])
        epsilon = 1e-6

        def log_density_at(parameter):
            return family.log_density(x, family.natural_from_expectation(parameter))

        numerical = []
        for coordinate in range(family.dim):
            step = np.zeros(family.dim)
            step[coordinate] = epsilon
            numerical.append((log_density_at(expectation + step) - log_density_at(expectation - step)) / (2.0 * epsilon))

        np.testing.assert_allclose(family.dual_score(x, expectation), np.array(numerical), rtol=1e-5, atol=1e-5)


class MultivariateGaussianCoordinateTests(unittest.TestCase):
    def test_natural_and_expectation_coordinates_round_trip(self):
        family = MultivariateGaussianCoordinate(2)
        mean = np.array([0.5, -0.2])
        covariance = np.array([[1.4, 0.2], [0.2, 0.9]])
        expectation = np.concatenate((mean, (covariance + mean[:, None] * mean[None, :]).reshape(-1)))

        natural = family.natural_from_expectation(expectation)
        recovered = family.expectation_from_natural(natural)

        np.testing.assert_allclose(recovered, expectation)

    def test_log_density_matches_normal_density(self):
        family = MultivariateGaussianCoordinate(2)
        mean = np.array([0.5, -0.2])
        covariance = np.array([[1.4, 0.2], [0.2, 0.9]])
        expectation = np.concatenate((mean, (covariance + mean[:, None] * mean[None, :]).reshape(-1)))
        natural = family.natural_from_expectation(expectation)
        x = np.array([[0.5, -0.2], [1.0, 0.1]])

        log_density = family.log_density(x, natural)
        sign, logdet = np.linalg.slogdet(covariance)
        residual = x - mean
        expected = (
            -0.5 * 2 * np.log(2.0 * np.pi)
            - 0.5 * logdet
            - 0.5 * np.einsum("ni,ij,nj->n", residual, np.linalg.inv(covariance), residual)
        )

        self.assertEqual(sign, 1.0)
        np.testing.assert_allclose(log_density, expected)

    def test_dual_score_matches_finite_difference_on_independent_coordinates(self):
        family = MultivariateGaussianCoordinate(2)
        mean = np.array([0.2, -0.1])
        covariance = np.array([[1.5, 0.1], [0.1, 1.1]])
        expectation = np.concatenate((mean, (covariance + mean[:, None] * mean[None, :]).reshape(-1)))
        x = np.array([0.5, -0.4])
        epsilon = 1e-6

        def log_density_at(parameter):
            return family.log_density(x, family.natural_from_expectation(parameter))

        score = family.dual_score(x, expectation)
        for coordinate in (0, 1, 2, 5):
            step = np.zeros(family.dim)
            step[coordinate] = epsilon
            numerical = (log_density_at(expectation + step) - log_density_at(expectation - step)) / (2.0 * epsilon)
            np.testing.assert_allclose(score[coordinate], numerical, rtol=1e-5, atol=1e-5)

    def test_vectorized_dual_score_supports_class_blocks(self):
        family = MultivariateGaussianCoordinate(2)
        expectation = family.initial_expectation()
        expectations = np.stack((expectation, expectation))
        x = np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 0.25]])

        score = family.dual_score(x, expectations)

        self.assertEqual(score.shape, (3, 2, family.dim))
        self.assertTrue(np.all(np.isfinite(score)))


class PoissonCoordinateTests(unittest.TestCase):
    def test_natural_and_expectation_coordinates_round_trip(self):
        family = PoissonCoordinate()
        expectation = np.array([2.5])

        natural = family.natural_from_expectation(expectation)
        recovered = family.expectation_from_natural(natural)

        np.testing.assert_allclose(recovered, expectation)

    def test_log_density_matches_poisson_formula(self):
        family = PoissonCoordinate()
        expectation = np.array([3.0])
        natural = family.natural_from_expectation(expectation)

        log_density = family.log_density(np.array([0.0, 1.0, 2.0]), natural)
        expected = np.array([-3.0, np.log(3.0) - 3.0, 2.0 * np.log(3.0) - 3.0 - np.log(2.0)])

        np.testing.assert_allclose(log_density, expected)

    def test_log_density_is_stable_for_underflowing_rate(self):
        family = PoissonCoordinate()
        log_density = family.log_density(np.array([0.0, 1.0]), np.array([-1000.0]))

        np.testing.assert_allclose(log_density, np.array([0.0, -1000.0]))

    def test_dual_score_matches_finite_difference(self):
        family = PoissonCoordinate()
        x = 4.0
        expectation = np.array([2.0])
        epsilon = 1e-6

        def log_density_at(rate):
            return family.log_density(x, family.natural_from_expectation(np.array([rate])))

        numerical = (log_density_at(expectation[0] + epsilon) - log_density_at(expectation[0] - epsilon)) / (
            2.0 * epsilon
        )

        np.testing.assert_allclose(family.dual_score(x, expectation), np.array([numerical]), rtol=1e-6)


class ExponentialMeanCoordinateTests(unittest.TestCase):
    def test_natural_and_expectation_coordinates_round_trip(self):
        family = ExponentialMeanCoordinate()
        expectation = np.array([2.5])

        natural = family.natural_from_expectation(expectation)
        recovered = family.expectation_from_natural(natural)

        np.testing.assert_allclose(recovered, expectation)

    def test_log_density_matches_exponential_formula(self):
        family = ExponentialMeanCoordinate()
        expectation = np.array([2.0])
        natural = family.natural_from_expectation(expectation)

        log_density = family.log_density(np.array([0.0, 2.0, 4.0]), natural)
        expected = -np.log(2.0) - np.array([0.0, 2.0, 4.0]) / 2.0

        np.testing.assert_allclose(log_density, expected)

    def test_project_natural_keeps_parameter_in_domain(self):
        family = ExponentialMeanCoordinate(natural_margin=1e-6)

        projected = family.project_natural(np.array([1.0, 0.0, -2.0]))

        np.testing.assert_allclose(projected, np.array([-1e-6, -1e-6, -2.0]))

    def test_dual_score_matches_finite_difference(self):
        family = ExponentialMeanCoordinate()
        x = 3.0
        expectation = np.array([2.0])
        epsilon = 1e-6

        def log_density_at(mean):
            return family.log_density(x, family.natural_from_expectation(np.array([mean])))

        numerical = (log_density_at(expectation[0] + epsilon) - log_density_at(expectation[0] - epsilon)) / (
            2.0 * epsilon
        )

        np.testing.assert_allclose(family.dual_score(x, expectation), np.array([numerical]), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
