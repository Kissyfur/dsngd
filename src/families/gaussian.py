import numpy as np

from src.families.base import ExponentialFamilyCoordinate


class GaussianKnownVarianceCoordinate(ExponentialFamilyCoordinate):
    """Univariate Gaussian coordinate with fixed variance and mean expectation parameter."""

    def __init__(self, variance=1.0):
        if variance <= 0.0:
            raise ValueError("variance must be positive")
        super().__init__(1)
        self.variance = float(variance)
        self.std = np.sqrt(self.variance)
        self.log_normalizer = -0.5 * np.log(2.0 * np.pi * self.variance)

    def sufficient_statistic(self, x):
        x = np.asarray(x, dtype=float)
        return np.expand_dims(x, axis=-1)

    def log_partition(self, natural_parameter):
        theta = np.asarray(natural_parameter, dtype=float)
        return 0.5 * self.variance * theta[..., 0] * theta[..., 0]

    def log_density(self, x, natural_parameter):
        x = np.asarray(x, dtype=float)
        theta = np.asarray(natural_parameter, dtype=float)
        centered = x - self.variance * theta[..., 0]
        return self.log_normalizer - 0.5 * centered * centered / self.variance

    def expectation_from_natural(self, natural_parameter):
        theta = np.asarray(natural_parameter, dtype=float)
        return self.variance * theta

    def natural_from_expectation(self, expectation_parameter):
        expectation = np.asarray(expectation_parameter, dtype=float)
        if expectation.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {expectation.shape[-1]}")
        return expectation / self.variance

    def dual_score(self, x, expectation_parameter):
        expectation = np.asarray(expectation_parameter, dtype=float)
        if expectation.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {expectation.shape[-1]}")
        x = np.asarray(x, dtype=float)
        if expectation.ndim == 1:
            return np.expand_dims((x - expectation[0]) / self.variance, axis=-1)
        return np.expand_dims((x[..., None] - expectation[..., 0]) / self.variance, axis=-1)

    def initial_expectation(self):
        return np.zeros(self.dim, dtype=float)

    def sample(self, expectation_parameter, rng, size=None):
        expectation = np.asarray(expectation_parameter, dtype=float)
        if expectation.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {expectation.shape[-1]}")
        return rng.normal(loc=expectation[..., 0], scale=self.std, size=size)
