import numpy as np

from src.families.base import ExponentialFamilyCoordinate


class MultivariateGaussianCoordinate(ExponentialFamilyCoordinate):
    """Full-covariance Gaussian block with mean and second moment expectation parameters."""

    def __init__(self, event_dim, min_variance=1e-10, natural_margin=1e-8):
        event_dim = int(event_dim)
        if event_dim <= 0:
            raise ValueError("event_dim must be positive")
        super().__init__(event_dim + event_dim * event_dim, input_dim=event_dim)
        self.event_dim = event_dim
        self.min_variance = float(min_variance)
        self.natural_margin = float(natural_margin)

    def sufficient_statistic(self, x):
        x = self._validate_observation(x)
        outer = x[..., :, None] * x[..., None, :]
        return np.concatenate(
            (x, outer.reshape(x.shape[:-1] + (self.event_dim * self.event_dim,))),
            axis=-1,
        )

    def log_partition(self, natural_parameter):
        theta = self._validate_parameter(natural_parameter)
        h, a = self._split_natural(theta)
        sign, logdet = np.linalg.slogdet(-a)
        if np.any(sign <= 0.0):
            raise ValueError("Gaussian quadratic natural matrix must be negative definite")
        a_inv = np.linalg.inv(a)
        quadratic = np.einsum("...i,...ij,...j->...", h, a_inv, h)
        return -0.25 * quadratic + 0.5 * self.event_dim * np.log(np.pi) - 0.5 * logdet

    def log_density(self, x, natural_parameter):
        statistics = self.sufficient_statistic(x)
        theta = self._validate_parameter(natural_parameter)
        return np.tensordot(statistics, theta, axes=([-1], [-1])) - self.log_partition(theta)

    def expectation_from_natural(self, natural_parameter):
        theta = self._validate_parameter(natural_parameter)
        h, a = self._split_natural(theta)
        sign, _ = np.linalg.slogdet(-a)
        if np.any(sign <= 0.0):
            raise ValueError("Gaussian quadratic natural matrix must be negative definite")
        covariance = -0.5 * np.linalg.inv(a)
        mean = np.einsum("...ij,...j->...i", covariance, h)
        second = covariance + mean[..., :, None] * mean[..., None, :]
        return self._join_expectation(mean, second)

    def natural_from_expectation(self, expectation_parameter):
        mean, covariance = self._mean_and_covariance(expectation_parameter)
        precision = np.linalg.inv(covariance)
        h = np.einsum("...ij,...j->...i", precision, mean)
        a = -0.5 * precision
        return self._join_natural(h, a)

    def dual_score(self, x, expectation_parameter):
        x = self._validate_observation(x)
        mean, covariance = self._mean_and_covariance(expectation_parameter)
        precision = np.linalg.inv(covariance)
        mean_ndim = mean.ndim - 1
        x_broadcast = x[(...,) + (None,) * mean_ndim + (slice(None),)]
        residual = x_broadcast - mean
        precision_residual = np.einsum("...ij,...j->...i", precision, residual)
        quadratic_score = 0.5 * (
            precision_residual[..., :, None] * precision_residual[..., None, :] - precision
        )
        mean_score = precision_residual - 2.0 * np.einsum("...ij,...j->...i", quadratic_score, mean)
        return np.concatenate(
            (
                mean_score,
                quadratic_score.reshape(quadratic_score.shape[:-2] + (self.event_dim * self.event_dim,)),
            ),
            axis=-1,
        )

    def initial_expectation(self):
        mean = np.zeros(self.event_dim, dtype=float)
        second = np.eye(self.event_dim, dtype=float)
        return self._join_expectation(mean, second)

    def sample(self, expectation_parameter, rng, size=None):
        mean, covariance = self._mean_and_covariance(expectation_parameter)
        if mean.ndim != 1:
            raise ValueError("sample expects one multivariate Gaussian expectation parameter")
        return rng.multivariate_normal(mean, covariance, size=size)

    def project_natural(self, natural_parameter):
        theta = self._validate_parameter(natural_parameter)
        h, a = self._split_natural(theta)
        projected_precision_half = self._floor_spd(-a, floor=self.natural_margin)
        return self._join_natural(h, -projected_precision_half)

    def _split_natural(self, natural_parameter):
        theta = self._validate_parameter(natural_parameter)
        h = theta[..., : self.event_dim]
        a = theta[..., self.event_dim :].reshape(theta.shape[:-1] + (self.event_dim, self.event_dim))
        return h, self._symmetrize(a)

    def _join_natural(self, h, a):
        return np.concatenate(
            (h, a.reshape(a.shape[:-2] + (self.event_dim * self.event_dim,))),
            axis=-1,
        )

    def _mean_and_covariance(self, expectation_parameter):
        expectation = self._validate_parameter(expectation_parameter)
        mean = expectation[..., : self.event_dim]
        second = expectation[..., self.event_dim :].reshape(
            expectation.shape[:-1] + (self.event_dim, self.event_dim)
        )
        second = self._symmetrize(second)
        covariance = second - mean[..., :, None] * mean[..., None, :]
        eigenvalues = np.linalg.eigvalsh(covariance)
        if np.any(eigenvalues <= self.min_variance):
            raise ValueError("Gaussian covariance must be positive definite")
        return mean, covariance

    def _join_expectation(self, mean, second):
        return np.concatenate(
            (mean, second.reshape(second.shape[:-2] + (self.event_dim * self.event_dim,))),
            axis=-1,
        )

    def _validate_observation(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            raise ValueError(f"expected observations with last dimension {self.event_dim}")
        if x.shape[-1] != self.event_dim:
            raise ValueError(f"expected observations with last dimension {self.event_dim}, got {x.shape[-1]}")
        return x

    def _validate_parameter(self, parameter):
        parameter = np.asarray(parameter, dtype=float)
        if parameter.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {parameter.shape[-1]}")
        return parameter

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + np.swapaxes(matrix, -1, -2))

    @staticmethod
    def _floor_spd(matrix, floor):
        matrix = MultivariateGaussianCoordinate._symmetrize(matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, floor)
        return np.matmul(eigenvectors * eigenvalues[..., None, :], np.swapaxes(eigenvectors, -1, -2))
