import unittest

import numpy as np

from src.experiments.ef_grid import (
    FAMILY_EXPERIMENT_SPECS,
    build_model,
    build_true_model,
    collect_sample,
    samples_seen,
    validation_nll,
)


class EFGridExperimentTests(unittest.TestCase):
    def test_pure_family_specs_have_three_complexity_levels(self):
        for family_name in ("categorical", "gaussian", "poisson", "exponential"):
            with self.subTest(family=family_name):
                spec = FAMILY_EXPERIMENT_SPECS[family_name]
                self.assertEqual(len(spec.complexity_scenarios), 3)
                self.assertEqual([scenario[0] for scenario in spec.complexity_scenarios], ["M1", "M2", "M3"])

    def test_family_specs_build_valid_true_models_and_samples(self):
        for family_name in ("categorical", "gaussian", "poisson", "exponential"):
            with self.subTest(family=family_name):
                spec = FAMILY_EXPERIMENT_SPECS[family_name]
                _, many_classes, family_factories = spec.complexity_scenarios[0]
                model = build_model(many_classes, family_factories)
                true_model = build_true_model(many_classes, family_factories, sigma=0.1, seed=3)
                x, y = collect_sample(true_model, size=8, batch=4, seed=5)

                self.assertEqual(x.shape, (8, len(model.families)))
                self.assertEqual(y.shape, (8,))
                self.assertTrue(np.all((0 <= y) & (y < many_classes)))
                self.assertTrue(np.isfinite(validation_nll(true_model, true_model.eta, x, y)))

    def test_samples_seen_matches_kept_optimizer_history(self):
        x_axis = samples_seen(train_size=1000, batch=100, iter_keep=4)

        np.testing.assert_array_equal(x_axis, np.array([0, 200, 400, 600, 800, 1000]))


if __name__ == "__main__":
    unittest.main()
