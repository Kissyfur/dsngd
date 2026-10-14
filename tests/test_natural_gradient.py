import unittest

import numpy as np

from src.algorithms.dsngd import DSNGD_JointMLR
from src.algorithms.sgd import SGD_JointMLR
from src.model.joint_mlr import JointMLR
from src.model.statistic import (ExponentialStatistic, GaussianMeanVarianceStatistic, GaussianStatistic,
                                 PoissonStatistic)

MODELS = {
    'discrete': [2, 4, 3],
    'gaussian': [GaussianStatistic(variance=1.5), GaussianStatistic()],
    'gaussian-mv': [GaussianMeanVarianceStatistic()],
    'poisson': [PoissonStatistic(), PoissonStatistic()],
    'exponential': [ExponentialStatistic(), ExponentialStatistic()],
    'mixed': [3, GaussianMeanVarianceStatistic(), PoissonStatistic(), ExponentialStatistic(), GaussianStatistic()],
}


def flatten(eta):
    return np.concatenate([eta[0], eta[1].reshape(-1)])


def log_partition_gradient(model, eta):
    # grad F(eta) in natural coordinates: p(y) for y < s and p(y) grad F_r(beta_{r,y})
    p = model.class_probabilities(eta)
    blocks = []
    for cs, rows, _ in model.T.slices():
        expectation = cs.expectation(eta[1][rows])
        blocks.append(expectation[:cs.dim] * p)
    return flatten((p[:-1], np.vstack(blocks)))


def fisher(model, eta, h=1e-6):
    shape = eta[1].shape
    point = flatten(eta)
    columns = []
    for i in range(len(point)):
        e = np.zeros(len(point))
        e[i] = h
        plus, minus = point + e, point - e
        columns.append((log_partition_gradient(model, (plus[:len(eta[0])], plus[len(eta[0]):].reshape(shape)))
                        - log_partition_gradient(model, (minus[:len(eta[0])], minus[len(eta[0]):].reshape(shape))))
                       / (2 * h))
    hessian = np.column_stack(columns)
    return 0.5 * (hessian + hessian.T)


class NaturalGradientTests(unittest.TestCase):
    def test_dsngd_with_exact_dual_is_the_natural_gradient(self):
        # With zeta = eta, the DSNGD direction is G^-1 grad l (exact natural gradient)
        for name, features in MODELS.items():
            with self.subTest(model=name):
                model = JointMLR(3, features)
                np.random.seed(0)
                model.set_random_eta(0.8)
                x, y = model.sample(4, np.random.default_rng(1))

                gradient = flatten(SGD_JointMLR(model).gradient_log_conditional_probability((x, y), model.eta))
                natural_gradient = np.linalg.solve(fisher(model, model.eta), gradient)
                direction = flatten(DSNGD_JointMLR(model).aprox_natural_gradient_log_conditional_probability(
                    (x, y), model.eta, model.to_dual(model.eta)))

                error = np.linalg.norm(direction - natural_gradient) / np.linalg.norm(natural_gradient)
                self.assertLess(error, 1e-5)

    def test_max_entropy_dual_of_mixed_features(self):
        model = JointMLR(2, [4, GaussianMeanVarianceStatistic(), ExponentialStatistic()])
        alpha_dual, beta_dual = DSNGD_JointMLR(model).max_entropy_dual_parameter()
        psi = (model.T.dim + 1) * 2
        np.testing.assert_allclose(alpha_dual, [psi / 2, psi / 2])
        np.testing.assert_allclose(beta_dual[:, 0], psi / 2 * np.array([.25, .25, .25, .25, 0., 1., 1.]))


if __name__ == '__main__':
    unittest.main()
