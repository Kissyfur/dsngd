import unittest

import numpy as np
from scipy.special import softmax

from src.algorithms.adaGrad import AdaGrad_JointMLR
from src.algorithms.dsngd import DSNGD_JointMLR
from src.algorithms.sgd import SGD_JointMLR
from src.algorithms.sngd import SNGD_JointMLR
from src.data.sample_creator import JointMLRSampleIterator
from src.model.joint_mlr import JointMLR
from src.model.statistic import (ExponentialStatistic, GaussianMeanVarianceStatistic, GaussianStatistic,
                                 PoissonStatistic)


def mixed_features():
    return [4, GaussianMeanVarianceStatistic(), PoissonStatistic(), ExponentialStatistic(), GaussianStatistic()]


def random_model(s, m, sigma, seed):
    model = JointMLR(s, m)
    np.random.seed(seed)
    model.set_random_eta(sigma)
    return model


class JointMLRContinuousTests(unittest.TestCase):
    def test_sample_iterator_gives_valid_mixed_batches(self):
        problem = random_model(3, mixed_features(), 1., seed=0)
        batches = list(JointMLRSampleIterator(problem, epoch_length=10, epochs=2, batch=4, random_seed=1))

        self.assertEqual([len(y) for _, y in batches], [4, 4, 2, 4, 4, 2])
        x = np.vstack([b[0] for b in batches])
        y = np.concatenate([b[1] for b in batches])
        self.assertEqual(x.shape, (20, 5))
        self.assertTrue(np.all((0 <= y) & (y < 3)))
        self.assertTrue(np.all(np.isin(x[:, 0], np.arange(4))))
        self.assertTrue(np.all((x[:, 2] >= 0) & (x[:, 2] == np.floor(x[:, 2]))))
        self.assertTrue(np.all(x[:, 3] > 0))

    def test_random_eta_is_valid_and_keeps_class_priors(self):
        sigma = 0.8
        problem = random_model(4, [GaussianMeanVarianceStatistic(), PoissonStatistic(), ExponentialStatistic()],
                               sigma, seed=3)
        alpha, beta = problem.eta
        self.assertTrue(np.all(beta[1] < 0) and np.all(beta[3] < 0))
        np.random.seed(3)
        drawn_alpha = np.random.normal(0., sigma, size=3)
        np.testing.assert_allclose(problem.class_probabilities(problem.eta), softmax(np.append(drawn_alpha, 0.)))

    def test_to_dual_matches_enumeration_for_discrete_features(self):
        model = random_model(3, [2, 4, 3], 0.9, seed=4)
        pyx = model.compute_all_pyx(model.eta)
        identity = model.T.identity(np.array(list(model.compute_all_x())))
        alpha_dual, beta_dual = model.to_dual(model.eta)
        np.testing.assert_allclose(alpha_dual, np.sum(pyx, axis=0), rtol=1e-12)
        np.testing.assert_allclose(beta_dual, np.dot(identity.T, pyx), rtol=1e-12)

    def test_to_dual_matches_sample_averages_for_mixed_features(self):
        model = random_model(3, mixed_features(), 1., seed=5)
        x, y = model.sample(400000, np.random.default_rng(6))
        alpha_dual, beta_dual = model.to_dual(model.eta)
        y_identity = model.S.identity(y)
        np.testing.assert_allclose(np.mean(y_identity, axis=0), alpha_dual, atol=5e-3)
        np.testing.assert_allclose(np.dot(model.T.identity(x).T, y_identity) / len(y), beta_dual, rtol=0.03, atol=5e-3)

    def test_kl_metric_is_zero_at_true_model_and_positive_elsewhere(self):
        problem = random_model(3, mixed_features(), 1., seed=7)
        model = JointMLR(3, mixed_features())
        kl = model.compute_metrics(problem, [problem.eta, model.eta])
        self.assertAlmostEqual(kl[0], 0., places=12)
        self.assertGreater(kl[1], 0.)
        self.assertEqual(problem.x_support.shape, (100000, 5))

    def test_short_fits_on_mixed_features_reduce_kl(self):
        problem = random_model(3, mixed_features(), 1., seed=8)
        sample = JointMLRSampleIterator(problem, 20000, 1, 100, random_seed=9)
        for algo_class, lr in ((SGD_JointMLR, (0.01, 0.01)), (AdaGrad_JointMLR, 0.1),
                               (DSNGD_JointMLR, (0.001, 0.01)), (SNGD_JointMLR, (0.001, 0.01))):
            with self.subTest(algorithm=algo_class.CLASS_NAME):
                model = JointMLR(3, mixed_features())
                model.fit(sample, algo_class(model), lr=lr, iter_keep=10)
                model.compute_history_metrics(problem)
                error = model.history['error']
                self.assertTrue(np.all(np.isfinite(error)))
                self.assertLess(error[-1], 0.5 * error[0])


if __name__ == '__main__':
    unittest.main()
