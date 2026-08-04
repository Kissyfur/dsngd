import numpy as np

from src.families.base import ExponentialFamilyCoordinate


class GaussianUnknownVarianceCoordinate(ExponentialFamilyCoordinate):
    """Univariate Gaussian coordinate with mean and second moment as expectation parameters."""

    def __init__(self, min_variance=1e-10, natural_margin=1e-8):
        super().__init__(2)
        self.min_variance = float(min_variance)
        self.natural_margin = float(natural_margin)

    def sufficient_statistic(self, x):
        x = np.asarray(x, dtype=float)
        return np.stack((x, x * x), axis=-1)

    def log_partition(self, natural_parameter):
        theta = self._validate_natural(natural_parameter)
        theta_1 = theta[..., 0]
        theta_2 = theta[..., 1]
        if np.any(theta_2 >= 0.0):
            raise ValueError("Gaussian quadratic natural parameter must be negative")
        return -theta_1 * theta_1 / (4.0 * theta_2) + 0.5 * np.log(-np.pi / theta_2)

    def log_density(self, x, natural_parameter):
        x = np.asarray(x, dtype=float)
        theta = self._validate_natural(natural_parameter)
        parameter_shape = theta.shape[:-1]
        x_broadcast = x[(...,) + (None,) * len(parameter_shape)]
        return (
            theta[..., 0] * x_broadcast
            + theta[..., 1] * x_broadcast * x_broadcast
            - self.log_partition(theta)
        )

    def expectation_from_natural(self, natural_parameter):
        theta = self._validate_natural(natural_parameter)
        theta_1 = theta[..., 0]
        theta_2 = theta[..., 1]
        if np.any(theta_2 >= 0.0):
            raise ValueError("Gaussian quadratic natural parameter must be negative")
        mean = -theta_1 / (2.0 * theta_2)
        variance = -0.5 / theta_2
        return np.stack((mean, mean * mean + variance), axis=-1)

    def natural_from_expectation(self, expectation_parameter):
        mean, variance = self._mean_and_variance(expectation_parameter)
        return np.stack((mean / variance, -0.5 / variance), axis=-1)

    def dual_score(self, x, expectation_parameter):
        mean, variance = self._mean_and_variance(expectation_parameter)
        x = np.asarray(x, dtype=float)
        x_broadcast = x[(...,) + (None,) * mean.ndim]
        residual = x_broadcast - mean
        grad_second = -0.5 / variance + 0.5 * residual * residual / (variance * variance)
        grad_mean = residual / variance - 2.0 * mean * grad_second
        return np.stack((grad_mean, grad_second), axis=-1)

    def initial_expectation(self):
        return np.array([0.0, 1.0], dtype=float)

    def sample(self, expectation_parameter, rng, size=None):
        mean, variance = self._mean_and_variance(expectation_parameter)
        return rng.normal(loc=mean, scale=np.sqrt(variance), size=size)

    def project_natural(self, natural_parameter):
        theta = self._validate_natural(natural_parameter)
        projected = theta.copy()
        projected[..., 1] = np.minimum(projected[..., 1], -self.natural_margin)
        return projected

    def _mean_and_variance(self, expectation_parameter):
        expectation = np.asarray(expectation_parameter, dtype=float)
        if expectation.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {expectation.shape[-1]}")
        mean = expectation[..., 0]
        second_moment = expectation[..., 1]
        variance = second_moment - mean * mean
        if np.any(variance <= self.min_variance):
            raise ValueError("Gaussian variance must be positive")
        return mean, variance

    def _validate_natural(self, natural_parameter):
        theta = np.asarray(natural_parameter, dtype=float)
        if theta.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {theta.shape[-1]}")
        return theta
