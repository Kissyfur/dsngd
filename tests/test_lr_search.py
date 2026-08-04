import contextlib
import io
import unittest

import numpy as np

from src.algorithms import LineSearch
from src.algorithms.adagrad_ef import AdaGrad_NaiveBayesEF
from src.algorithms.sgd_ef import SGD_NaiveBayesEF
from src.data.ef_sample_creator import NaiveBayesEFSampleIterator
from src.families import CategoricalCoordinate, GaussianKnownVarianceCoordinate
from src.model.naive_bayes_ef import NaiveBayesEF


def build_model():
    return NaiveBayesEF(2, [CategoricalCoordinate(2), GaussianKnownVarianceCoordinate(variance=1.0)])


def validation_nll(model, eta, x, y):
    log_probabilities = model.log_conditional_probabilities(x, eta)
    return -float(np.mean(log_probabilities[np.arange(len(y)), y]))


class LearningRateSearchTests(unittest.TestCase):
    def test_base_line_search_handles_large_iter_keep(self):
        optimizer = LineSearch(lambda _obs, param: np.zeros_like(param))
        sample = [0, 1, 2]

        etas = optimizer.run(sample, np.array([1.0]), lr=np.array([0.1, 1.0]), iter_keep=100)

        self.assertEqual(len(etas), len(sample))

    def test_adjust_lr_with_data_works_for_generic_ef_optimizers(self):
        true_model = build_model()
        true_model.set_eta((np.array([0.0]), [np.array([[0.3, -0.1]]), np.array([[-0.5, 0.5]])]))
        x_val, y_val = next(iter(NaiveBayesEFSampleIterator(true_model, epoch_length=6, batch=6, random_seed=9)))
        optimizer = SGD_NaiveBayesEF(build_model())

        data = {
            "model_factory": build_model,
            "sample_factory": lambda: NaiveBayesEFSampleIterator(
                true_model,
                epoch_length=6,
                batch=3,
                random_seed=4,
            ),
            "validation_curve": lambda fit_model, etas: np.array(
                [validation_nll(fit_model, eta, x_val, y_val) for eta in etas]
            ),
            "score_tail": 2,
        }

        with contextlib.redirect_stdout(io.StringIO()):
            lr = optimizer.adjust_lr_with_data(data, progress_bar=False)

        self.assertEqual(len(lr), 2)
        self.assertTrue(np.all(np.isfinite(lr)))
        self.assertTrue(np.all(np.asarray(lr) > 0.0))

    def test_adjust_lr_with_data_supports_adagrad_ef(self):
        true_model = build_model()
        true_model.set_eta((np.array([0.0]), [np.array([[0.3, -0.1]]), np.array([[-0.5, 0.5]])]))
        x_val, y_val = next(iter(NaiveBayesEFSampleIterator(true_model, epoch_length=6, batch=6, random_seed=9)))
        optimizer = AdaGrad_NaiveBayesEF(build_model())

        data = {
            "model_factory": build_model,
            "sample_factory": lambda: NaiveBayesEFSampleIterator(
                true_model,
                epoch_length=6,
                batch=3,
                random_seed=4,
            ),
            "validation_curve": lambda fit_model, etas: np.array(
                [validation_nll(fit_model, eta, x_val, y_val) for eta in etas]
            ),
            "score_tail": 2,
        }

        with contextlib.redirect_stdout(io.StringIO()):
            lr = optimizer.adjust_lr_with_data(data, progress_bar=False)

        lr_values = np.asarray(lr).reshape(-1)
        self.assertEqual(len(lr_values), 1)
        self.assertTrue(np.all(np.isfinite(lr_values)))
        self.assertGreater(lr_values[0], 0.0)


if __name__ == "__main__":
    unittest.main()
