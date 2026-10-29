import unittest

import numpy as np

from src.algorithms.dsngd import DSNGD_JointMLR
from src.algorithms.sgd import SGD_JointMLR
from src.data.sample_creator import JointMLRSampleIterator
from src.model.joint_mlr import JointMLR
from src.model.statistic import (BetaStatistic, DirichletStatistic, ExponentialStatistic, GammaStatistic,
                                 GaussianMeanVarianceStatistic, GaussianStatistic, MultivariateGaussianStatistic,
                                 NaturalPointStatistic, PoissonStatistic, VonMisesStatistic)

MODELS = {
    'discrete': [2, 4, 3],
    'gaussian': [GaussianStatistic(variance=1.5), GaussianStatistic()],
    'gaussian-mv': [GaussianMeanVarianceStatistic()],
    'poisson': [PoissonStatistic(), PoissonStatistic()],
    'exponential': [ExponentialStatistic(), ExponentialStatistic()],
    'multivariate-gaussian': [MultivariateGaussianStatistic(2), MultivariateGaussianStatistic(1)],
    'gamma': [GammaStatistic(), GammaStatistic()],
    'beta': [BetaStatistic(), BetaStatistic()],
    'dirichlet': [DirichletStatistic(3)],
    'von-mises': [VonMisesStatistic(), VonMisesStatistic()],
    'mixed': [3, GaussianMeanVarianceStatistic(), PoissonStatistic(), ExponentialStatistic(), GaussianStatistic(),
              MultivariateGaussianStatistic(2), GammaStatistic(), BetaStatistic(), DirichletStatistic(3),
              VonMisesStatistic()],
    # Families repeated in non contiguous features, which are computed together
    'repeated': [3, GaussianMeanVarianceStatistic(), GammaStatistic(), 3, GaussianMeanVarianceStatistic(),
                 MultivariateGaussianStatistic(2), GammaStatistic(), MultivariateGaussianStatistic(2), 2],
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

    def test_points_in_natural_coordinates_follow_the_map_sequence(self):
        # Along the dual parameters of a run, the auxiliary points of the Gamma, Beta, Dirichlet and von Mises features,
        # kept in natural coordinates, approach those of the MAP sequence, and the others are those of the MAP sequence
        model = JointMLR(3, MODELS['mixed'])
        np.random.seed(2)
        model.set_random_eta(0.8)
        dsngd = DSNGD_JointMLR(model)
        dual = dsngd.max_entropy_dual_parameter()
        points = dsngd.auxiliary_points(dual)
        for x, y in JointMLRSampleIterator(model, 20000, 1, 100, random_seed=3):
            id_y = model.S.identity(y)
            dual = (dual[0] + np.sum(id_y, axis=0), dual[1] + np.dot(model.T.identity(x).T, id_y))
            points = dsngd.auxiliary_points(dual, points)
        for group, point, map_point in zip(model.T.groups, points, dsngd.auxiliary_points(dual)):
            with self.subTest(statistic=group.statistic):
                if isinstance(group.statistic, NaturalPointStatistic):
                    np.testing.assert_allclose(point[0], map_point[0], rtol=1e-3, atol=1e-4)
                else:
                    np.testing.assert_array_equal(point, map_point)

    def test_max_entropy_dual_of_mixed_features(self):
        model = JointMLR(2, [4, GaussianMeanVarianceStatistic(), ExponentialStatistic()])
        alpha_dual, beta_dual = DSNGD_JointMLR(model).max_entropy_dual_parameter()
        psi = (model.T.dim + 1) * 2
        np.testing.assert_allclose(alpha_dual, [psi / 2, psi / 2])
        np.testing.assert_allclose(beta_dual[:, 0], psi / 2 * np.array([.25, .25, .25, .25, 0., 1., 1.]))


if __name__ == '__main__':
    unittest.main()
