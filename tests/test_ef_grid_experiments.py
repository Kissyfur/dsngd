from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import src.experiments.ef_grid as ef_grid
from src.experiments.ef_grid import (
    ALGORITHMS,
    build_model,
    build_true_model,
    collect_sample,
    learning_rate_columns,
    samples_seen,
    validation_nll,
)
from src.experiments.ef_specs import EFExperimentSpec, FAMILY_EXPERIMENT_SPECS, PURE_FAMILY_KEYS, gaussian


class EFGridExperimentTests(unittest.TestCase):
    def test_pure_family_specs_have_three_complexity_levels(self):
        self.assertIn("multivariate_gaussian", PURE_FAMILY_KEYS)
        for family_name in PURE_FAMILY_KEYS:
            with self.subTest(family=family_name):
                spec = FAMILY_EXPERIMENT_SPECS[family_name]
                self.assertEqual(len(spec.complexity_scenarios), 3)
                self.assertEqual([scenario[0] for scenario in spec.complexity_scenarios], ["M1", "M2", "M3"])

    def test_multivariate_gaussian_spec_uses_one_block_with_growing_event_dimension(self):
        spec = FAMILY_EXPERIMENT_SPECS["multivariate_gaussian"]
        event_dims = []

        for _, _, family_factories in spec.complexity_scenarios:
            self.assertEqual(len(family_factories), 1)
            event_dims.append(family_factories[0]().event_dim)

        self.assertEqual(event_dims, [3, 5, 6])

    def test_scalar_continuous_specs_use_comparable_variable_counts(self):
        categorical_spec = FAMILY_EXPERIMENT_SPECS["categorical"]
        target_dimensions = [
            build_model(many_classes, family_factories).parameter_dim
            for _, many_classes, family_factories in categorical_spec.complexity_scenarios
        ]

        for family_name in ("gaussian", "poisson", "exponential"):
            with self.subTest(family=family_name):
                spec = FAMILY_EXPERIMENT_SPECS[family_name]
                feature_counts = [len(family_factories) for _, _, family_factories in spec.complexity_scenarios]
                parameter_dimensions = [
                    build_model(many_classes, family_factories).parameter_dim
                    for _, many_classes, family_factories in spec.complexity_scenarios
                ]

                self.assertEqual(feature_counts, [13, 26, 39])
                self.assertEqual(parameter_dimensions, target_dimensions)

    def test_mixed_repeated_spec_repeats_all_family_types(self):
        spec = FAMILY_EXPERIMENT_SPECS["mixed_repeated"]

        self.assertEqual([scenario[0] for scenario in spec.complexity_scenarios], ["M1", "M2", "M3"])
        self.assertEqual([scenario[1] for scenario in spec.complexity_scenarios], [10, 20, 30])
        self.assertEqual([len(scenario[2]) for scenario in spec.complexity_scenarios], [4, 8, 12])

    def test_grid_algorithms_include_adagrad(self):
        self.assertEqual([name for name, _ in ALGORITHMS], ["SGD", "AdaGrad", "DSNGD"])

    def test_learning_rate_columns_supports_single_parameter_schedules(self):
        self.assertEqual(learning_rate_columns(np.array([0.1])), (0.1, ""))

    def test_family_specs_build_valid_true_models_and_samples(self):
        for family_name in PURE_FAMILY_KEYS + ("mixed_repeated",):
            with self.subTest(family=family_name):
                spec = FAMILY_EXPERIMENT_SPECS[family_name]
                _, many_classes, family_factories = spec.complexity_scenarios[0]
                model = build_model(many_classes, family_factories)
                true_model = build_true_model(many_classes, family_factories, sigma=0.1, seed=3)
                x, y = collect_sample(true_model, size=8, batch=4, seed=5)

                self.assertEqual(x.shape, (8, model.observation_dim))
                self.assertEqual(y.shape, (8,))
                self.assertTrue(np.all((0 <= y) & (y < many_classes)))
                self.assertTrue(np.isfinite(validation_nll(true_model, true_model.eta, x, y)))

    def test_samples_seen_matches_kept_optimizer_history(self):
        x_axis = samples_seen(train_size=1000, batch=100, iter_keep=4)

        np.testing.assert_array_equal(x_axis, np.array([0, 200, 400, 600, 800, 1000]))

    def test_grid_experiment_uses_independent_lr_and_evaluation_samples(self):
        captured = {}

        def fake_collect_sample(_model, size, batch, seed):
            return np.array([[float(seed)]]), np.array([size + batch])

        def fake_choose_best_lr(
            _algorithm_class,
            _model_factory,
            _true_model,
            _lr_train_size,
            _batch,
            train_seed,
            x_lr_val,
            y_lr_val,
            progress_bar=True,
        ):
            captured["lr_train_seed"] = train_seed
            captured["lr_validation_seed"] = int(x_lr_val[0, 0])
            captured["lr_validation_size_marker"] = int(y_lr_val[0])
            return np.array([1.0, 1.0])

        def fake_run_algorithm(
            _algorithm_class,
            _model_factory,
            _true_model,
            _lr,
            train_size,
            batch,
            seed,
            x_eval,
            y_eval,
            true_nll,
            progress_bar=True,
        ):
            captured["final_train_seed"] = seed
            captured["evaluation_seed"] = int(x_eval[0, 0])
            captured["evaluation_size_marker"] = int(y_eval[0])
            return np.ones(len(samples_seen(train_size, batch)))

        spec = EFExperimentSpec(
            key="test",
            title="Test",
            output_name="test",
            default_output_dir="test",
            complexity_scenarios=(("M1", 2, (gaussian(),)),),
        )

        with patch.object(ef_grid, "ENTROPY_SCENARIOS", (("Entropy", 0.1),)), \
            patch.object(ef_grid, "ALGORITHMS", (("ALG", object),)), \
            patch.object(ef_grid, "build_true_model", return_value=SimpleNamespace(eta=None)), \
            patch.object(ef_grid, "collect_sample", side_effect=fake_collect_sample), \
            patch.object(ef_grid, "validation_nll", return_value=0.0), \
            patch.object(ef_grid, "choose_best_lr", side_effect=fake_choose_best_lr), \
            patch.object(ef_grid, "run_algorithm", side_effect=fake_run_algorithm), \
            patch.object(ef_grid, "plot_grid"), \
            patch.object(ef_grid, "save_summary"), \
            patch.object(ef_grid, "save_curves"), \
            patch.object(ef_grid.logging, "info"), \
            patch("builtins.print"):
            ef_grid.run_grid_experiment(
                spec,
                output_dir="unused",
                train_size=100,
                batch=10,
                lr_size=20,
                lr_validation_size=30,
                eval_validation_size=100_000,
                many_experiments=1,
                progress_bar=False,
            )

        self.assertEqual(captured["lr_validation_size_marker"], 1030)
        self.assertEqual(captured["evaluation_size_marker"], 101000)
        self.assertEqual(captured["lr_train_seed"], captured["final_train_seed"])
        self.assertNotEqual(captured["lr_validation_seed"], captured["evaluation_seed"])

    def test_grid_experiment_rejects_lr_search_longer_than_training(self):
        spec = EFExperimentSpec(
            key="test",
            title="Test",
            output_name="test",
            default_output_dir="test",
            complexity_scenarios=(("M1", 2, (gaussian(),)),),
        )

        with self.assertRaisesRegex(ValueError, "training prefix"):
            ef_grid.run_grid_experiment(
                spec,
                output_dir="unused",
                train_size=100,
                lr_size=101,
                progress_bar=False,
            )


if __name__ == "__main__":
    unittest.main()
