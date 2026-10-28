import unittest

import numpy as np

from src.model.statistic import (BetaStatistic, CanonicalFeatureStatistic, CanonicalStatistic, DirichletStatistic,
                                 ExponentialStatistic, GammaStatistic, GaussianMeanVarianceStatistic,
                                 GaussianStatistic, MultivariateGaussianStatistic, PoissonStatistic)

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


class CanonicalStatisticTests(unittest.TestCase):
    def test_expectation_and_samples(self):
        statistic = CanonicalStatistic(4)
        theta = np.array([0.5, -1., 0.2])
        probabilities = statistic.expectation(theta)
        np.testing.assert_allclose(probabilities, np.exp(np.append(theta, 0.)) / np.sum(np.exp(np.append(theta, 0.))))
        values = statistic.sample(np.tile(theta[:, None], (1, 100000)), np.random.default_rng(2))
        np.testing.assert_allclose(np.bincount(values, minlength=4) / len(values), probabilities, atol=0.01)

    def test_dsngd_terms_match_generic_formula_with_categorical_score(self):
        # g_j = 1{x=j} / p_j - 1{x=K} / p_K (Table 1) inserted in the generic Naive Bayes formula
        rng = np.random.default_rng(3)
        many_values, many_classes = 4, 3
        statistic = CanonicalStatistic(many_values)
        alpha_dual = rng.uniform(1., 3., size=many_classes)
        dual = rng.dirichlet(np.ones(many_values), size=many_classes).T * alpha_dual
        x = np.array([0, 3, 1, 3, 2, 0])
        q_minus_e = rng.normal(size=(len(x), many_classes))

        block, alpha_term = statistic.dsngd_terms(x, dual, alpha_dual, q_minus_e)

        p = dual / alpha_dual
        theta_star = p[:-1]
        g = (np.eye(many_values)[x].T[:-1, :, None] / theta_star[:, None, :]
             - (x == many_values - 1)[None, :, None] / p[-1])
        u_q = q_minus_e * np.sum(alpha_dual) / alpha_dual
        np.testing.assert_allclose(block, np.sum(g * u_q, axis=1))
        w = 1. - np.sum(theta_star[:, None, :] * g, axis=0)
        np.testing.assert_allclose(alpha_term, np.sum(w * u_q, axis=0))


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


if __name__ == '__main__':
    unittest.main()
