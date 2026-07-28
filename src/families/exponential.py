import numpy as np

from src.families.base import ExponentialFamilyCoordinate


class ExponentialMeanCoordinate(ExponentialFamilyCoordinate):
    """Exponential coordinate with mean as expectation parameter."""

    def __init__(self, min_mean=1e-12, natural_margin=1e-8):
        super().__init__(1)
        self.min_mean = float(min_mean)
        self.natural_margin = float(natural_margin)

    def sufficient_statistic(self, x):
        x = self._validate_observation(x)
        return np.expand_dims(x, axis=-1)

    def log_partition(self, natural_parameter):
        theta = np.asarray(natural_parameter, dtype=float)
        if np.any(theta >= 0.0):
            raise ValueError("exponential natural parameter must be negative")
        return -np.log(-theta[..., 0])

    def log_density(self, x, natural_parameter):
        x = self._validate_observation(x)
        theta = np.asarray(natural_parameter, dtype=float)
        if np.any(theta >= 0.0):
            raise ValueError("exponential natural parameter must be negative")
        return np.log(-theta[..., 0]) + x * theta[..., 0]

    def expectation_from_natural(self, natural_parameter):
        theta = np.asarray(natural_parameter, dtype=float)
        if np.any(theta >= 0.0):
            raise ValueError("exponential natural parameter must be negative")
        return -1.0 / theta

    def natural_from_expectation(self, expectation_parameter):
        mean = self._validate_expectation(expectation_parameter)
        return -1.0 / mean

    def dual_score(self, x, expectation_parameter):
        x = self._validate_observation(x)
        mean = self._validate_expectation(expectation_parameter)
        if mean.ndim == 1:
            return np.expand_dims(x / (mean[0] * mean[0]) - 1.0 / mean[0], axis=-1)
        mean_value = mean[..., 0]
        return np.expand_dims(x[..., None] / (mean_value * mean_value) - 1.0 / mean_value, axis=-1)

    def initial_expectation(self):
        return np.ones(self.dim, dtype=float)

    def sample(self, expectation_parameter, rng, size=None):
        mean = self._validate_expectation(expectation_parameter)
        return rng.exponential(scale=mean[..., 0], size=size)

    def project_natural(self, natural_parameter):
        theta = np.asarray(natural_parameter, dtype=float)
        return np.minimum(theta, -self.natural_margin)

    def _validate_expectation(self, expectation_parameter):
        expectation = np.asarray(expectation_parameter, dtype=float)
        if expectation.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {expectation.shape[-1]}")
        if np.any(expectation <= self.min_mean):
            raise ValueError("exponential mean must be positive")
        return expectation

    @staticmethod
    def _validate_observation(x):
        x = np.asarray(x, dtype=float)
        if np.any(x < 0.0):
            raise ValueError("exponential observations must be non-negative")
        return x
