import unittest

import numpy as np

from src.families import CategoricalCoordinate, MultivariateGaussianCoordinate
from src.model.joint_mlr import JointMLR
from src.model.naive_bayes_ef import NaiveBayesEF


class NaiveBayesEFTests(unittest.TestCase):
    def test_categorical_model_matches_joint_mlr_posteriors(self):
        many_classes = 3
        feature_values = [2, 3]
        old_model = JointMLR(many_classes, feature_values)
        new_model = NaiveBayesEF(
            many_classes,
            [CategoricalCoordinate(value) for value in feature_values],
        )

        alpha = np.array([0.1, -0.2])
        beta = np.array(
            [
                [0.3, -0.2, 0.1],
                [0.5, 0.0, -0.3],
                [-0.1, 0.4, 0.2],
            ]
        )
        old_model.set_eta((alpha, beta))
        new_model.set_eta((alpha, [beta[:1], beta[1:]]))

        x = np.array(list(old_model.compute_all_x()))

        np.testing.assert_allclose(
            new_model.conditional_probabilities(x),
            old_model.conditional_probabilities(x, [old_model.eta])[0],
        )

    def test_class_probabilities_and_dual_shapes(self):
        model = NaiveBayesEF(4, [CategoricalCoordinate(2), CategoricalCoordinate(3)])

        class_probabilities = model.class_probabilities()
        alpha_dual, beta_dual_blocks = model.to_dual()

        np.testing.assert_allclose(np.sum(class_probabilities), 1.0)
        np.testing.assert_allclose(alpha_dual, class_probabilities)
        self.assertEqual(len(beta_dual_blocks), 2)
        self.assertEqual(beta_dual_blocks[0].shape, (1, 4))
        self.assertEqual(beta_dual_blocks[1].shape, (2, 4))
        self.assertTrue(np.all(beta_dual_blocks[0] > 0.0))
        self.assertTrue(np.all(beta_dual_blocks[1] > 0.0))

    def test_multivariate_family_has_separate_observation_and_parameter_dimensions(self):
        family = MultivariateGaussianCoordinate(3)
        model = NaiveBayesEF(2, [CategoricalCoordinate(2), family])

        self.assertEqual(model.observation_dim, 4)
        self.assertEqual(model.feature_dim, 1 + family.dim)
        self.assertEqual(model.parameter_dim, model.alpha.size + model.many_classes * model.feature_dim)


if __name__ == "__main__":
    unittest.main()
