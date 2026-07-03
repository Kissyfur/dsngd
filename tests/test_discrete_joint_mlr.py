import unittest

import numpy as np

from src.algorithms.dsngd import DSNGD_JointMLR
from src.data.sample_creator import JointMLRSampleIterator
from src.model.joint_mlr import JointMLR


class JointMLRDiscreteSmokeTests(unittest.TestCase):
    def test_conditional_probabilities_are_normalized(self):
        model = JointMLR(3, [2, 3])
        np.random.seed(7)
        model.set_random_eta(0.25)

        x = np.array(list(model.compute_all_x()))
        probabilities = model.conditional_probabilities(x, [model.eta])[0]

        self.assertEqual(probabilities.shape, (len(x), model.S.many_values))
        np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
        self.assertTrue(np.all(probabilities > 0.0))

    def test_sample_iterator_returns_valid_batches(self):
        problem = JointMLR(3, [2, 3])
        np.random.seed(11)
        problem.set_random_eta(0.5)

        sample = JointMLRSampleIterator(problem, epoch_length=12, epochs=1, batch=4, random_seed=3)
        x, y = next(iter(sample))

        self.assertEqual(x.shape, (4, 2))
        self.assertEqual(y.shape, (4,))
        self.assertTrue(np.all((0 <= y) & (y < problem.S.many_values)))
        self.assertTrue(np.all((0 <= x[:, 0]) & (x[:, 0] < 2)))
        self.assertTrue(np.all((0 <= x[:, 1]) & (x[:, 1] < 3)))

    def test_dsngd_direction_has_expected_shapes(self):
        model = JointMLR(3, [2, 3])
        optimizer = DSNGD_JointMLR(model)
        sample = (
            np.array([[0, 0], [1, 2], [0, 1], [1, 0]]),
            np.array([0, 1, 2, 0]),
        )

        ng_alpha, ng_beta = optimizer.aprox_natural_gradient_log_conditional_probability(
            sample,
            model.eta,
            optimizer.max_entropy_dual_parameter(),
        )

        self.assertEqual(ng_alpha.shape, model.alpha.shape)
        self.assertEqual(ng_beta.shape, model.beta.shape)
        self.assertTrue(np.all(np.isfinite(ng_alpha)))
        self.assertTrue(np.all(np.isfinite(ng_beta)))

    def test_short_dsngd_fit_updates_model_and_history(self):
        problem = JointMLR(3, [2, 3])
        np.random.seed(5)
        problem.set_random_eta(0.4)

        model = JointMLR(3, [2, 3])
        optimizer = DSNGD_JointMLR(model)
        sample = JointMLRSampleIterator(problem, epoch_length=20, epochs=1, batch=5, random_seed=2)

        model.fit(sample, optimizer, lr=[0.01, 0.0], iter_keep=2)

        self.assertIn("etas", model.history)
        self.assertGreaterEqual(len(model.history["etas"]), 2)
        self.assertTrue(np.all(np.isfinite(model.alpha)))
        self.assertTrue(np.all(np.isfinite(model.beta)))
        self.assertGreater(np.linalg.norm(model.alpha) + np.linalg.norm(model.beta), 0.0)


if __name__ == "__main__":
    unittest.main()
