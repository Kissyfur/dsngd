import unittest

import numpy as np

from src.algorithms.dsngd import DSNGD_JointMLR
from src.algorithms.dsngd_ef import DSNGD_NaiveBayesEF, EmpiricalSufficientStatisticDual
from src.families import (
    CategoricalCoordinate,
    ExponentialMeanCoordinate,
    GaussianKnownVarianceCoordinate,
    MultivariateGaussianCoordinate,
)
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


def split_dual_beta(beta_dual, feature_values):
    blocks = []
    start = 0
    for many_values in feature_values:
        end = start + many_values
        blocks.append(beta_dual[start:start + many_values - 1])
        start = end
    return blocks


class TrackingDualParametrization:
    def __init__(self):
        self.default = EmpiricalSufficientStatisticDual()
        self.initial_calls = 0
        self.update_calls = 0

    def initial_parameter(self, model, strength=None):
        self.initial_calls += 1
        return self.default.initial_parameter(model, strength=strength)

    def update(self, model, dual_parameter, sample, all_categorical=False):
        self.update_calls += 1
        return self.default.update(model, dual_parameter, sample, all_categorical=all_categorical)


class DSNGDEFTests(unittest.TestCase):
    def test_categorical_direction_matches_legacy_dsngd(self):
        many_classes = 3
        feature_values = [2, 3]
        old_model = JointMLR(many_classes, feature_values)
        new_model = NaiveBayesEF(
            many_classes,
            [CategoricalCoordinate(value) for value in feature_values],
        )
        old_optimizer = DSNGD_JointMLR(old_model)
        new_optimizer = DSNGD_NaiveBayesEF(new_model)

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

        old_dual = old_optimizer.max_entropy_dual_parameter()
        new_dual = (old_dual[0], split_dual_beta(old_dual[1], feature_values))

        old_alpha_direction, old_beta_direction = old_optimizer.aprox_natural_gradient_log_conditional_probability(
            sample,
            old_model.eta,
            old_dual,
        )
        new_alpha_direction, new_beta_directions = new_optimizer.approx_natural_gradient_log_conditional_probability(
            sample,
            new_model.eta,
            new_dual,
        )

        np.testing.assert_allclose(new_alpha_direction, old_alpha_direction)
        np.testing.assert_allclose(np.vstack(new_beta_directions), old_beta_direction)

    def test_default_categorical_dual_matches_legacy_initialization(self):
        many_classes = 4
        feature_values = [3, 5, 2]
        old_optimizer = DSNGD_JointMLR(JointMLR(many_classes, feature_values))
        new_optimizer = DSNGD_NaiveBayesEF(
            NaiveBayesEF(many_classes, [CategoricalCoordinate(value) for value in feature_values])
        )

        old_class_dual, old_beta_dual = old_optimizer.max_entropy_dual_parameter()
        new_class_dual, new_beta_dual_blocks = new_optimizer.max_entropy_dual_parameter()

        np.testing.assert_allclose(new_class_dual, old_class_dual)
        np.testing.assert_allclose(np.vstack(new_beta_dual_blocks), np.vstack(split_dual_beta(old_beta_dual, feature_values)))

    def test_short_generic_run_updates_parameters(self):
        model = NaiveBayesEF(3, [CategoricalCoordinate(2), CategoricalCoordinate(3)])
        optimizer = DSNGD_NaiveBayesEF(model)
        sample = [
            (
                np.array([[0, 0], [1, 2], [0, 1], [1, 0]]),
                np.array([0, 1, 2, 0]),
            ),
            (
                np.array([[1, 1], [0, 2], [1, 0], [0, 1]]),
                np.array([1, 2, 0, 1]),
            ),
        ]

        etas = optimizer.run(sample, model.eta, lr=[0.01, 0.0], iter_keep=2)

        self.assertGreaterEqual(len(etas), 2)
        final_alpha, final_beta_blocks = etas[-1]
        self.assertTrue(np.all(np.isfinite(final_alpha)))
        for block in final_beta_blocks:
            self.assertTrue(np.all(np.isfinite(block)))
        self.assertGreater(
            np.linalg.norm(final_alpha) + sum(np.linalg.norm(block) for block in final_beta_blocks),
            0.0,
        )

    def test_generic_branch_runs_on_mixed_continuous_blocks(self):
        model = NaiveBayesEF(
            2,
            [
                CategoricalCoordinate(2),
                GaussianKnownVarianceCoordinate(variance=1.0),
                ExponentialMeanCoordinate(),
                MultivariateGaussianCoordinate(2),
            ],
        )
        optimizer = DSNGD_NaiveBayesEF(model)
        sample = [
            (
                np.array(
                    [
                        [0, -0.5, 1.0, 0.0, 0.0],
                        [1, 0.5, 2.0, 1.0, -1.0],
                        [0, 0.2, 0.75, 0.5, 0.25],
                        [1, -1.0, 1.5, -0.5, 0.5],
                    ],
                    dtype=float,
                ),
                np.array([0, 1, 0, 1]),
            )
        ]

        etas = optimizer.run(sample, model.eta, lr=[0.001, 0.0], iter_keep=1)

        final_alpha, final_beta_blocks = etas[-1]
        self.assertTrue(np.all(np.isfinite(final_alpha)))
        for block in final_beta_blocks:
            self.assertTrue(np.all(np.isfinite(block)))
        self.assertGreater(
            np.linalg.norm(final_alpha) + sum(np.linalg.norm(block) for block in final_beta_blocks),
            0.0,
        )

    def test_run_uses_injected_dual_parametrization(self):
        model = NaiveBayesEF(2, [CategoricalCoordinate(2)])
        dual_parametrization = TrackingDualParametrization()
        optimizer = DSNGD_NaiveBayesEF(model, dual_parametrization=dual_parametrization)
        sample = [
            (
                np.array([[0], [1], [0], [1]]),
                np.array([0, 1, 0, 1]),
            )
        ]

        optimizer.run(sample, model.eta, lr=[0.01, 0.0], iter_keep=1)

        self.assertEqual(dual_parametrization.initial_calls, 1)
        self.assertEqual(dual_parametrization.update_calls, 1)

    def test_clone_preserves_injected_dual_parametrization(self):
        model = NaiveBayesEF(2, [CategoricalCoordinate(2)])
        dual_parametrization = TrackingDualParametrization()
        optimizer = DSNGD_NaiveBayesEF(model, dual_parametrization=dual_parametrization)

        clone = optimizer.clone_for_model(NaiveBayesEF(2, [CategoricalCoordinate(2)]))

        self.assertIs(clone.dual_parametrization, dual_parametrization)


if __name__ == "__main__":
    unittest.main()
