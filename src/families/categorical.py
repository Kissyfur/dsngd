import numpy as np
from scipy.special import logsumexp

from src.families.base import ExponentialFamilyCoordinate


class CategoricalCoordinate(ExponentialFamilyCoordinate):
    """Categorical coordinate with the last category as canonical baseline."""

    def __init__(self, many_values, min_probability=1e-12):
        if many_values < 2:
            raise ValueError("many_values must be at least 2")
        super().__init__(many_values - 1)
        self.many_values = int(many_values)
        self.min_probability = float(min_probability)

    def sufficient_statistic(self, x):
        x = np.asarray(x, dtype=int)
        if np.any((x < 0) | (x >= self.many_values)):
            raise ValueError("categorical values must be in [0, many_values)")

        statistic = np.zeros(x.shape + (self.dim,), dtype=float)
        mask = x < self.dim
        if np.any(mask):
            statistic[mask, x[mask]] = 1.0
        return statistic

    def log_partition(self, natural_parameter):
        theta = np.asarray(natural_parameter, dtype=float)
        zeros = np.zeros(theta.shape[:-1] + (1,), dtype=float)
        return logsumexp(np.concatenate([theta, zeros], axis=-1), axis=-1)

    def log_density(self, x, natural_parameter):
        statistic = self.sufficient_statistic(x)
        theta = np.asarray(natural_parameter, dtype=float)
        return np.sum(statistic * theta, axis=-1) - self.log_partition(theta)

    def expectation_from_natural(self, natural_parameter):
        theta = np.asarray(natural_parameter, dtype=float)
        logits = np.concatenate([theta, np.zeros(theta.shape[:-1] + (1,), dtype=float)], axis=-1)
        probabilities = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
        return probabilities[..., :-1]

    def natural_from_expectation(self, expectation_parameter):
        probabilities = self._full_probabilities(expectation_parameter)
        return np.log(probabilities[..., :-1]) - np.log(probabilities[..., -1:])

    def dual_score(self, x, expectation_parameter):
        x = np.asarray(x, dtype=int)
        probabilities = self._full_probabilities(expectation_parameter)
        score = np.zeros(x.shape + (self.dim,), dtype=float)
        non_baseline = x < self.dim
        if np.any(non_baseline):
            score[non_baseline, x[non_baseline]] = 1.0 / probabilities[..., x[non_baseline]]
        baseline = x == self.dim
        if np.any(baseline):
            score[baseline] = -1.0 / probabilities[..., -1:]
        return score

    def initial_expectation(self):
        return np.ones(self.dim, dtype=float) / self.many_values

    def sample(self, expectation_parameter, rng, size=None):
        probabilities = self._full_probabilities(expectation_parameter)
        return rng.choice(self.many_values, size=size, p=probabilities)

    def _full_probabilities(self, expectation_parameter):
        expectation = np.asarray(expectation_parameter, dtype=float)
        if expectation.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {expectation.shape[-1]}")
        baseline = 1.0 - np.sum(expectation, axis=-1, keepdims=True)
        probabilities = np.concatenate([expectation, baseline], axis=-1)
        if np.any(probabilities <= self.min_probability):
            raise ValueError("expectation_parameter must be in the categorical simplex interior")
        return probabilities
