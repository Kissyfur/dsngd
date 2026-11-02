import unittest

import numpy as np

from src.model.statistic import (BetaStatistic, CanonicalFeatureStatistic, CanonicalStatistic, DirichletStatistic,
                                 ExponentialStatistic, GammaStatistic, GaussianMeanVarianceStatistic,
                                 GaussianStatistic, MultivariateGaussianStatistic, PoissonStatistic, Statistic,
                                 VonMisesStatistic)

MULTIVARIATE_GAUSSIAN = MultivariateGaussianStatistic(2)

# family, natural parameter and observations
FAMILIES = {
    'gaussian': (GaussianStatistic(variance=1.7), np.array([0.4]), np.array([-1., 0., 1., 2.5])),
    'gaussian-mv': (GaussianMeanVarianceStatistic(), np.array([0.6, -0.8]), np.array([-1., 0., 1., 2.5])),
    'multivariate-gaussian': (MULTIVARIATE_GAUSSIAN,
                              MULTIVARIATE_GAUSSIAN.random_natural(np.array([[.3], [-.2], [.2], [.4], [-.1]]))[:, 0],
                              np.array([[0., 0.], [1., -1.], [-.5, 2.], [2., .3]])),
    'poisson': (PoissonStatistic(), np.array([0.7]), np.array([0., 1., 2., 5.])),
    'exponential': (ExponentialStatistic(), np.array([-1.3]), np.array([.1, .5, 1., 3.])),
    'gamma': (GammaStatistic(), np.array([-1.5, 0.7]), np.array([.1, .5, 1., 3.])),
    'beta': (BetaStatistic(), np.array([0.5, 1.2]), np.array([.05, .3, .5, .9])),
    'von-mises': (VonMisesStatistic(), np.array([0.8, -0.5]), np.array([-2., 0., 1., 3.])),
    'dirichlet': (DirichletStatistic(3), np.array([0.5, 1.2, -0.3]),
                  np.array([[.2, .3, .5], [.6, .3, .1], [.1, .1, .8], [1 / 3, 1 / 3, 1 / 3]])),
}


def finite_difference_jacobian(f, theta, h=1e-6):
    columns = []
    for i in range(len(theta)):
        e = np.zeros(len(theta))
        e[i] = h
        columns.append((np.atleast_1d(f(theta + e)) - np.atleast_1d(f(theta - e))) / (2 * h))
    return np.column_stack(columns)


class ExponentialFamilyStatisticTests(unittest.TestCase):
    def test_expectation_is_gradient_of_log_partition(self):
        for name, (family, theta, _) in FAMILIES.items():
            with self.subTest(family=name):
                gradient = finite_difference_jacobian(family.log_partition, theta)[0]
                np.testing.assert_allclose(family.expectation(theta), gradient, rtol=1e-6, atol=1e-8)

    def test_dual_hessian_product_is_inverse_hessian_of_log_partition(self):
        # H(theta*) = hess F*(theta*) = (hess F(theta))^-1, for one and for several columns
        rng = np.random.default_rng(4)
        for name, (family, theta, _) in FAMILIES.items():
            with self.subTest(family=name):
                hessian = finite_difference_jacobian(family.expectation, theta)
                rho = rng.normal(size=(family.dim, 3))
                theta_star = np.tile(family.expectation(theta)[:, None], (1, 3))
                np.testing.assert_allclose(family.dual_hessian_product(theta_star, rho),
                                           np.linalg.solve(hessian, rho), rtol=1e-5, atol=1e-7)

    def test_dual_score_is_score_in_expectation_coordinates(self):
        # g(x, theta*) = (hess F(theta))^-1 (T(x) - theta*)
        for name, (family, theta, x) in FAMILIES.items():
            with self.subTest(family=name):
                theta_star = family.expectation(theta)
                hessian = finite_difference_jacobian(family.expectation, theta)
                expected = np.linalg.solve(hessian, (family.transform(x) - theta_star).T)
                score = family.dual_score(x, theta_star[:, None])
                self.assertEqual(score.shape, (family.dim, len(x), 1))
                np.testing.assert_allclose(score[:, :, 0], expected, rtol=1e-5, atol=1e-7)

    def test_samples_match_expectation(self):
        rng = np.random.default_rng(0)
        for name, (family, theta, _) in FAMILIES.items():
            with self.subTest(family=name):
                statistics = family.transform(family.sample(np.tile(theta[:, None], (1, 200000)), rng))
                standard_error = statistics.std(axis=0) / np.sqrt(len(statistics))
                np.testing.assert_array_less(np.abs(statistics.mean(axis=0) - family.expectation(theta)),
                                             5 * standard_error)

    def test_random_natural_parameters_are_valid(self):
        z = np.random.default_rng(1).normal(0., 2., size=(5, 50))
        for name, (family, _, _) in FAMILIES.items():
            with self.subTest(family=name):
                theta = family.random_natural(z[:family.dim])
                self.assertEqual(theta.shape, (family.dim, 50))
                self.assertTrue(np.all(np.isfinite(family.log_partition(theta))))
                self.assertTrue(np.all(np.isfinite(family.expectation(theta))))

    def test_expectation_to_natural_maps_invert_expectation(self):
        gamma = GammaStatistic()
        shape, rate = np.array([0.05, 0.5, 1., 7., 80.]), np.array([3., 0.2, 1., 0.5, 10.])
        estimated_shape, estimated_rate = gamma.shape_rate(gamma.expectation(np.stack([-rate, shape - 1.])))
        np.testing.assert_allclose(estimated_shape, shape, rtol=1e-8)
        np.testing.assert_allclose(estimated_rate, rate, rtol=1e-8)
        dirichlet = DirichletStatistic(3)
        a = np.array([[0.1, 1., 50., 2., 500., 0.01], [0.3, 1., 80., 0.05, 900., 0.02], [2., 1., 20., 0.5, 300., 5.]])
        np.testing.assert_allclose(dirichlet.parameters(dirichlet.expectation(a - 1.)), a, rtol=1e-8)
        von_mises = VonMisesStatistic()
        kappa = np.array([1e-6, 0.01, 0.5, 2., 30., 3000.])
        np.testing.assert_allclose(von_mises.concentration(von_mises.mean_resultant(kappa)), kappa, rtol=1e-6)


class CanonicalStatisticTests(unittest.TestCase):
    def test_expectation_and_samples(self):
        statistic = CanonicalStatistic(4)
        theta = np.array([0.5, -1., 0.2])
        probabilities = statistic.expectation(theta)
        np.testing.assert_allclose(probabilities, np.exp(np.append(theta, 0.)) / np.sum(np.exp(np.append(theta, 0.))))
        values = statistic.sample(np.tile(theta[:, None], (1, 100000)), np.random.default_rng(2))
        np.testing.assert_allclose(np.bincount(values, minlength=4) / len(values), probabilities, atol=0.01)

    def test_hessian_gives_the_categorical_score(self):
        # H(theta*) (T(x) - theta*) is the score g_j = 1{x=j} / p_j - 1{x=K} / p_K of Table 1
        many_values = 4
        statistic = CanonicalStatistic(many_values)
        p = np.random.default_rng(3).dirichlet(np.ones(many_values), size=3).T
        theta_star = p[:-1]
        x = np.array([0, 3, 1, 3, 2, 0])
        residual = statistic.transform(x).T[:, :, None] - theta_star[:, None, :]
        expected = (np.eye(many_values)[x].T[:-1, :, None] / theta_star[:, None, :]
                    - (x == many_values - 1)[None, :, None] / p[-1])
        np.testing.assert_allclose(statistic.dual_hessian_product(theta_star[:, None, :], residual), expected,
                                   atol=1e-12)

    def test_block_of_the_direction_without_products_by_the_hessian(self):
        # Proposition 3.3 summed over a mini-batch, N_i / p_i - N_m / p_m, is H(theta*) (M - a theta*) (Remark 4.12)
        rng = np.random.default_rng(5)
        statistic = CanonicalStatistic(4)
        alpha_dual = rng.uniform(5., 10., size=3)
        dual = rng.dirichlet(np.ones(4), size=(2, 3)).transpose(2, 0, 1) * alpha_dual
        values, w = rng.integers(0, 4, size=(6, 2)), rng.normal(size=(6, 3))
        m = np.stack([np.dot(np.eye(4)[values[:, j]].T, w) for j in range(2)], axis=1)
        a = np.sum(w, axis=0)
        for actual, expected in zip(statistic.dsngd_block(statistic.auxiliary_point(dual, alpha_dual), m, a),
                                    Statistic.dsngd_block(statistic, Statistic.auxiliary_point(statistic, dual,
                                                                                               alpha_dual), m, a)):
            np.testing.assert_allclose(actual, expected)


class NaturalPointTests(unittest.TestCase):
    """Gamma, Beta, Dirichlet and von Mises, whose auxiliary point is kept in natural coordinates."""
    names = ['gamma', 'beta', 'von-mises', 'dirichlet']

    def test_natural_point_gives_expectation_and_hessian(self):
        rng = np.random.default_rng(6)
        for name in self.names:
            family, theta, _ = FAMILIES[name]
            with self.subTest(family=name):
                theta_star, factors = family.natural_point(theta[:, None])
                np.testing.assert_allclose(theta_star[:, 0], family.expectation(theta), rtol=1e-12)
                np.testing.assert_allclose(family.natural(theta_star)[:, 0], theta, rtol=1e-8)
                rho = rng.normal(size=(family.dim, 1))
                hessian = finite_difference_jacobian(family.expectation, theta)
                np.testing.assert_allclose(family.hessian_product(factors, rho), np.linalg.solve(hessian, rho),
                                           rtol=1e-5, atol=1e-7)

    def test_auxiliary_point_follows_the_average(self):
        # The first point is at the average. One Newton step toward a moved average leaves an error of the order of
        # the square of the move, and a few steps reach averages far away.
        rng = np.random.default_rng(7)
        for name in self.names:
            family = FAMILIES[name][0]
            with self.subTest(family=name):
                theta = family.random_natural(rng.normal(0., 0.5, size=(family.dim, 3, 4)))
                alpha_dual = rng.uniform(50., 100., size=4)
                average = family.expectation(theta)
                point = family.auxiliary_point(average * alpha_dual, alpha_dual)
                np.testing.assert_allclose(point[0], average, rtol=1e-12, atol=1e-14)
                moved = family.expectation(theta + rng.normal(0., 1e-3, size=theta.shape))
                step = family.auxiliary_point(moved * alpha_dual, alpha_dual, point)
                self.assertLess(np.max(np.abs(step[0] - moved)), 0.02 * np.max(np.abs(moved - average)))
                for _ in range(10):
                    far = family.expectation(family.random_natural(rng.normal(0., 2., size=theta.shape)))
                    far_point = point
                    for _ in range(50):
                        far_point = family.auxiliary_point(far * alpha_dual, alpha_dual, far_point)
                    np.testing.assert_allclose(far_point[0], far, rtol=1e-10, atol=1e-12)

    def test_block_at_a_natural_point(self):
        # The block of Theorem 3.2 at a natural point is that of the general formula at its theta*
        rng = np.random.default_rng(8)
        for name in self.names:
            family = FAMILIES[name][0]
            with self.subTest(family=name):
                point = family.natural_point(family.random_natural(rng.normal(0., 0.5, size=(family.dim, 3, 4))))
                m, a = rng.normal(size=(family.dim, 3, 4)), rng.normal(size=4)
                for actual, expected in zip(family.dsngd_block(point, m, a),
                                            Statistic.dsngd_block(family, point[0], m, a)):
                    np.testing.assert_allclose(actual, expected, rtol=1e-10)


class ExactSpeedUpTests(unittest.TestCase):
    """The faster computations give the same numbers as the direct ones."""

    def test_samplers_draw_the_numbers_of_the_general_samplers(self):
        rng = np.random.default_rng(0)
        theta = np.stack([rng.normal(size=(3, 50)), -rng.uniform(.2, 2., size=(3, 50))])
        cases = [(GaussianStatistic(variance=1.7), theta[:1], lambda r: r.normal(1.7 * theta[0], np.sqrt(1.7))),
                 (GaussianMeanVarianceStatistic(), theta,
                  lambda r: r.normal(-0.5 / theta[1] * theta[0], np.sqrt(-0.5 / theta[1]))),
                 (ExponentialStatistic(), theta[1:], lambda r: r.exponential(-1. / theta[1]))]
        for statistic, parameters, general in cases:
            with self.subTest(statistic=statistic):
                np.testing.assert_array_equal(statistic.sample(parameters, np.random.default_rng(1)),
                                              general(np.random.default_rng(1)))
        gamma = np.random.default_rng(1).gamma(0.3 * theta + 1.)
        np.testing.assert_array_equal(DirichletStatistic(2).sample(0.3 * theta, np.random.default_rng(1)),
                                      np.moveaxis(gamma / np.sum(gamma, axis=0), 0, -1))

    def test_multivariate_gaussian_samples_of_each_class_and_dimension_one(self):
        rng = np.random.default_rng(2)
        classes = rng.integers(0, 4, size=40)
        for d in (1, 3):
            with self.subTest(d=d):
                statistic = MultivariateGaussianStatistic(d)
                theta = statistic.random_natural(rng.normal(size=(statistic.dim, 2, 4)) * 0.5)
                np.testing.assert_array_equal(statistic.sample_classes(theta, classes, np.random.default_rng(3)),
                                              statistic.sample(theta[..., classes], np.random.default_rng(3)))
        statistic, matrices = MultivariateGaussianStatistic(1), rng.uniform(.2, 3., size=(5, 1, 1))
        np.testing.assert_array_equal(statistic._inverse(matrices), np.linalg.inv(matrices))
        np.testing.assert_array_equal(statistic._cholesky(matrices), np.linalg.cholesky(matrices))

    def test_trigamma_is_polygamma(self):
        from scipy.special import polygamma
        from src.model.statistic import trigamma
        x = np.random.default_rng(4).uniform(1e-3, 50., size=1000)
        np.testing.assert_array_equal(trigamma(x), polygamma(1, x))


class CanonicalFeatureStatisticTests(unittest.TestCase):
    def test_mixed_features(self):
        T = CanonicalFeatureStatistic([3, GaussianMeanVarianceStatistic(), PoissonStatistic()])
        self.assertFalse(T.discrete)
        self.assertEqual((T.dim, T.dual_dim, T.continuous_dim), (5, 6, 3))
        x = np.array([[0, 0.5, 2.], [2, -1., 0.]])
        np.testing.assert_allclose(T.transform(x), [[1, 0, 0.5, 0.25, 2.], [0, 0, -1., 1., 0.]])
        np.testing.assert_allclose(T.identity(x), [[1, 0, 0, 0.5, 0.25, 2.], [0, 0, 1, -1., 1., 0.]])
        beta = np.tile(T.initial_natural()[:, None], (1, 2))
        np.testing.assert_allclose(T.expectation(beta)[:, 0], [1 / 3, 1 / 3, 1 / 3, 0., 1., 1.])

    def test_discrete_features_keep_previous_attributes(self):
        T = CanonicalFeatureStatistic([10, 5])
        self.assertTrue(T.discrete)
        self.assertEqual((T.dim, T.dual_dim, T.r, T.psi), (13, 15, 2, 50))

    def test_features_of_the_same_family_are_computed_together(self):
        # Features of a family that are not contiguous are computed together and placed in their rows and columns
        T = CanonicalFeatureStatistic([3, GaussianMeanVarianceStatistic(), PoissonStatistic(), 3,
                                       GaussianMeanVarianceStatistic(), 2])
        self.assertEqual(len(T.groups), 4)
        x = np.array([[0, 0.5, 2., 1, -1., 1], [2, -1., 0., 0, 3., 0]])
        np.testing.assert_allclose(T.transform(x), np.hstack([cs.transform(x[:, j])
                                                             for j, cs in enumerate(T.canonical_statistics)]))
        np.testing.assert_allclose(T.identity(x), np.hstack([cs.identity(x[:, j])
                                                            for j, cs in enumerate(T.canonical_statistics)]))
        np.testing.assert_allclose(T.identity(x)[:, T.expectation_rows], T.transform(x))
        beta = np.tile(T.initial_natural()[:, None], (1, 20000))
        sample = T.sample(beta, np.random.default_rng(1))
        np.testing.assert_allclose(np.mean(T.identity(sample), axis=0), T.expectation(beta[:, :1])[:, 0], atol=0.04)


if __name__ == '__main__':
    unittest.main()
