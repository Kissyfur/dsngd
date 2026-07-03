import unittest

import numpy as np

from src.algorithms.dsngd_ef import DSNGD_NaiveBayesEF
from src.data.ef_sample_creator import NaiveBayesEFSampleIterator
from src.families import CategoricalCoordinate, GaussianKnownVarianceCoordinate
from src.model.naive_bayes_ef import NaiveBayesEF


class ContinuousEFIntegrationTests(unittest.TestCase):
    def test_gaussian_model_posteriors_are_normalized(self):
        model = NaiveBayesEF(2, [GaussianKnownVarianceCoordinate(variance=1.5)])
        alpha = np.array([0.2])
        beta = [np.array([[-1.0, 1.0]])]
        model.set_eta((alpha, beta))

        probabilities = model.conditional_probabilities(np.array([[-2.0], [0.0], [2.0]]))

        self.assertEqual(probabilities.shape, (3, 2))
        np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
        self.assertTrue(np.all(probabilities > 0.0))

    def test_mixed_categorical_gaussian_dsngd_direction_is_finite(self):
        model = NaiveBayesEF(
            3,
            [
                CategoricalCoordinate(2),
                GaussianKnownVarianceCoordinate(variance=2.0),
            ],
        )
        optimizer = DSNGD_NaiveBayesEF(model)
        sample = (
            np.array([[0, -1.0], [1, 0.5], [0, 2.0], [1, -0.25]]),
            np.array([0, 1, 2, 1]),
        )

        alpha_direction, beta_directions = optimizer.approx_natural_gradient_log_conditional_probability(
            sample,
            model.eta,
            optimizer.max_entropy_dual_parameter(),
        )

        self.assertEqual(alpha_direction.shape, model.alpha.shape)
        self.assertEqual(beta_directions[0].shape, model.beta_blocks[0].shape)
        self.assertEqual(beta_directions[1].shape, model.beta_blocks[1].shape)
        self.assertTrue(np.all(np.isfinite(alpha_direction)))
        for direction in beta_directions:
            self.assertTrue(np.all(np.isfinite(direction)))

    def test_mixed_categorical_gaussian_run_updates_parameters(self):
        model = NaiveBayesEF(
            2,
            [
                CategoricalCoordinate(2),
                GaussianKnownVarianceCoordinate(variance=1.0),
            ],
        )
        optimizer = DSNGD_NaiveBayesEF(model)
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

    def test_mixed_sampler_returns_valid_batches(self):
        model = NaiveBayesEF(
            2,
            [
                CategoricalCoordinate(3),
                GaussianKnownVarianceCoordinate(variance=1.0),
            ],
        )
        sample = NaiveBayesEFSampleIterator(model, epoch_length=7, epochs=1, batch=4, random_seed=12)

        batches = list(sample)

        self.assertEqual(len(sample), 2)
        self.assertEqual(batches[0][0].shape, (4, 2))
        self.assertEqual(batches[1][0].shape, (3, 2))
        x = np.vstack([batch_x for batch_x, _ in batches])
        y = np.concatenate([batch_y for _, batch_y in batches])
        self.assertTrue(np.all((0 <= x[:, 0]) & (x[:, 0] < 3)))
        self.assertTrue(np.all(np.equal(np.mod(x[:, 0], 1.0), 0.0)))
        self.assertTrue(np.all((0 <= y) & (y < model.many_classes)))

    def test_generic_dsngd_can_run_on_synthetic_mixed_sample(self):
        true_model = NaiveBayesEF(
            2,
            [
                CategoricalCoordinate(2),
                GaussianKnownVarianceCoordinate(variance=1.0),
            ],
        )
        true_model.set_eta((np.array([0.1]), [np.array([[0.3, -0.1]]), np.array([[-0.5, 0.5]])]))
        fit_model = NaiveBayesEF(
            2,
            [
                CategoricalCoordinate(2),
                GaussianKnownVarianceCoordinate(variance=1.0),
            ],
        )
        optimizer = DSNGD_NaiveBayesEF(fit_model)
        sample = NaiveBayesEFSampleIterator(true_model, epoch_length=12, epochs=1, batch=4, random_seed=9)

        etas = optimizer.run(sample, fit_model.eta, lr=[0.005, 0.0], iter_keep=2)

        final_alpha, final_beta_blocks = etas[-1]
        self.assertTrue(np.all(np.isfinite(final_alpha)))
        for block in final_beta_blocks:
            self.assertTrue(np.all(np.isfinite(block)))


if __name__ == "__main__":
    unittest.main()
