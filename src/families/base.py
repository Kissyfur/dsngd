from abc import ABC, abstractmethod


class ExponentialFamilyCoordinate(ABC):
    """One coordinate of a Naive Bayes exponential-family feature vector."""

    def __init__(self, dim):
        self.dim = int(dim)

    @abstractmethod
    def sufficient_statistic(self, x):
        """Return the canonical sufficient statistic T(x)."""

    @abstractmethod
    def log_partition(self, natural_parameter):
        """Return F(theta) for the natural parameter theta."""

    @abstractmethod
    def log_density(self, x, natural_parameter):
        """Return log f(x; theta) in natural coordinates."""

    @abstractmethod
    def expectation_from_natural(self, natural_parameter):
        """Map natural coordinates theta to expectation coordinates theta_star."""

    @abstractmethod
    def natural_from_expectation(self, expectation_parameter):
        """Map expectation coordinates theta_star to natural coordinates theta."""

    @abstractmethod
    def dual_score(self, x, expectation_parameter):
        """Return grad_{theta_star} log f(x; theta_star)."""

    @abstractmethod
    def initial_expectation(self):
        """Return an interior expectation parameter for initialization."""

    @abstractmethod
    def sample(self, expectation_parameter, rng, size=None):
        """Draw samples using expectation coordinates."""

    def project_natural(self, natural_parameter):
        """Project a natural parameter back into this family's valid domain."""
        return natural_parameter
