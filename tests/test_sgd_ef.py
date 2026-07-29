import unittest

import numpy as np

from src.algorithms.sgd import SGD_JointMLR
from src.algorithms.adagrad_ef import AdaGrad_NaiveBayesEF
from src.algorithms.sgd_ef import SGD_NaiveBayesEF
from src.families import CategoricalCoordinate, ExponentialMeanCoordinate, GaussianKnownVarianceCoordinate
from src.model.joint_mlr import JointMLR
from src.model.naive_bayes_ef import NaiveBayesEF


def split_beta(beta, feature_values):
    blocks = []
    start = 0
    for many_values in feature_values:
        end = start + many_values - 1
        blocks.append(beta[start:end])
        start = end
    return blocks


class SGDEFTests(unittest.TestCase):
    def test_categorical_gradient_matches_legacy_sgd(self):
        many_classes = 3
        feature_values = [2, 3]
        old_model = JointMLR(many_classes, feature_values)
        new_model = NaiveBayesEF(
            many_classes,
            [CategoricalCoordinate(value) for value in feature_values],
        )
        old_optimizer = SGD_JointMLR(old_model)
        new_optimizer = SGD_NaiveBayesEF(new_model)

        alpha = np.array([0.1, -0.2])
        beta = np.array(
            [
                [0.3, -0.2, 0.1],
                [0.5, 0.0, -0.3],
                [-0.1, 0.4, 0.2],
            ]
        )
        old_model.set_eta((alpha, beta))
        new_model.set_eta((alpha, split_beta(beta, feature_values)))
        sample = (
            np.array([[0, 0], [1, 2], [0, 1], [1, 0], [1, 1]]),
            np.array([0, 1, 2, 0, 1]),
        )

        old_alpha_gradient, old_beta_gradient = old_optimizer.gradient_log_conditional_probability(
            sample,
            old_model.eta,
        )
        new_alpha_gradient, new_beta_gradients = new_optimizer.gradient_log_conditional_probability(
            sample,
            new_model.eta,
        )

        np.testing.assert_allclose(new_alpha_gradient, old_alpha_gradient)
        np.testing.assert_allclose(np.vstack(new_beta_gradients), old_beta_gradient)

    def test_short_generic_sgd_run_updates_parameters(self):
        model = NaiveBayesEF(
            2,
            [
                CategoricalCoordinate(2),
                GaussianKnownVarianceCoordinate(variance=1.0),
            ],
        )
        optimizer = SGD_NaiveBayesEF(model)
        sample = [
            (
                np.array([[0, -1.0], [1, 1.0], [0, -0.5], [1, 0.5]]),
                np.array([0, 1, 0, 1]),
            )
        ]

        etas = optimizer.run(sample, model.eta, lr=[0.01, 0.0], iter_keep=1)

        final_alpha, final_beta_blocks = etas[-1]
        self.assertTrue(np.all(np.isfinite(final_alpha)))
        for block in final_beta_blocks:
            self.assertTrue(np.all(np.isfinite(block)))
        self.assertGreater(
            np.linalg.norm(final_alpha) + sum(np.linalg.norm(block) for block in final_beta_blocks),
            0.0,
        )

    def test_run_projects_exponential_coordinate_after_large_step(self):
        model = NaiveBayesEF(2, [ExponentialMeanCoordinate(natural_margin=1e-6)])
        optimizer = SGD_NaiveBayesEF(model)
        sample = [
            (
                np.array([[10.0], [10.0]]),
                np.array([0, 0]),
            )
        ]

        etas = optimizer.run(sample, model.eta, lr=[1.0, 0.0], iter_keep=1)

        _, final_beta_blocks = etas[-1]
        self.assertTrue(np.all(final_beta_blocks[0] < 0.0))
        self.assertEqual(final_beta_blocks[0][0, 0], -1e-6)

    def test_short_generic_adagrad_run_updates_parameters(self):
        model = NaiveBayesEF(
            2,
            [
                CategoricalCoordinate(2),
                GaussianKnownVarianceCoordinate(variance=1.0),
            ],
        )
        optimizer = AdaGrad_NaiveBayesEF(model)
        sample = [
            (
                np.array([[0, -1.0], [1, 1.0], [0, -0.5], [1, 0.5]]),
                np.array([0, 1, 0, 1]),
            )
        ]

        etas = optimizer.run(sample, model.eta, lr=[0.01], iter_keep=1)

        final_alpha, final_beta_blocks = etas[-1]
        self.assertTrue(np.all(np.isfinite(final_alpha)))
        for block in final_beta_blocks:
            self.assertTrue(np.all(np.isfinite(block)))
        self.assertGreater(
            np.linalg.norm(final_alpha) + sum(np.linalg.norm(block) for block in final_beta_blocks),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
