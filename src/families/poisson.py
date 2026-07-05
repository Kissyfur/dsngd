import numpy as np
from scipy.special import gammaln

from src.families.base import ExponentialFamilyCoordinate


class PoissonCoordinate(ExponentialFamilyCoordinate):
    """Poisson coordinate with rate lambda as expectation parameter."""

    def __init__(self, min_rate=1e-12):
        super().__init__(1)
        self.min_rate = float(min_rate)

    def sufficient_statistic(self, x):
        x = self._validate_observation(x)
        return np.expand_dims(x, axis=-1)

    def log_partition(self, natural_parameter):
        theta = np.asarray(natural_parameter, dtype=float)
        return np.sum(np.exp(theta), axis=-1)

    def log_density(self, x, natural_parameter):
        x = self._validate_observation(x)
        theta = np.asarray(natural_parameter, dtype=float)
        return x * theta[..., 0] - np.exp(theta[..., 0]) - gammaln(x + 1.0)

    def expectation_from_natural(self, natural_parameter):
        theta = np.asarray(natural_parameter, dtype=float)
        return np.exp(theta)

    def natural_from_expectation(self, expectation_parameter):
        expectation = self._validate_expectation(expectation_parameter)
        return np.log(expectation)

    def dual_score(self, x, expectation_parameter):
        x = self._validate_observation(x)
        rate = self._validate_expectation(expectation_parameter)
        return np.expand_dims(x / rate[..., 0] - 1.0, axis=-1)

    def initial_expectation(self):
        return np.ones(self.dim, dtype=float)

    def sample(self, expectation_parameter, rng, size=None):
        rate = self._validate_expectation(expectation_parameter)
        return rng.poisson(lam=rate[..., 0], size=size)

    def _validate_expectation(self, expectation_parameter):
        expectation = np.asarray(expectation_parameter, dtype=float)
        if expectation.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {expectation.shape[-1]}")
        if np.any(expectation <= self.min_rate):
            raise ValueError("Poisson rate must be positive")
        return expectation

    @staticmethod
    def _validate_observation(x):
        x = np.asarray(x, dtype=float)
        if np.any(x < 0.0) or np.any(np.floor(x) != x):
            raise ValueError("Poisson observations must be non-negative integers")
        return x
