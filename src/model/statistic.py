import numpy as np
from scipy.special import logsumexp


class Statistic:
    discrete = False

    def __init__(self, dim):
        self.dim = dim
        self.dual_dim = dim

    def transform(self, rv):
        return rv

    def identity(self, rv):
        # Statistic accumulated by the DSNGD dual parameter
        return self.transform(rv)


class CanonicalStatistic(Statistic):
    discrete = True

    def __init__(self, many_values):
        self.many_values = many_values
        super().__init__(many_values - 1)
        self.dual_dim = many_values

    def identity(self, arr):
        arr = arr.copy().astype(int)
        id_arr = np.identity(self.many_values, dtype=int)
        id_arr = id_arr[arr]
        return id_arr

    def k(self, arr):
        arr = arr.copy().astype(int)
        id_arr = np.identity(self.many_values, dtype=int)
        id_arr[-1, :] = -1
        id_arr = id_arr[arr]
        return id_arr

    def canonical_statistic(self, arr):
        c_arr = self.identity(arr)
        return c_arr[:, :-1]

    def tau(self, arr):
        k_arr = self.k(arr)
        return k_arr[:, :-1]

    def transform(self, rv):
        return self.canonical_statistic(rv)

    def log_partition(self, theta):
        logits = np.concatenate([theta, np.zeros(theta.shape[:-1] + (1,))], axis=-1)
        return logsumexp(logits, axis=-1)

    def expectation(self, theta):
        # Probabilities of all the values, the expectation of identity
        logits = np.concatenate([theta, np.zeros(theta.shape[:-1] + (1,))], axis=-1)
        return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))

    def initial_natural(self):
        return np.zeros(self.dim)

    def random_natural(self, z):
        return z

    def sample(self, theta, rng):
        cumulative = np.cumsum(self.expectation(theta), axis=-1)
        values = np.sum(cumulative < rng.random((len(theta), 1)), axis=-1)
        return np.minimum(values, self.dim)

    def dsngd_terms(self, x, dual, alpha_dual, q_minus_e):
        x = x.astype(int)
        D = np.sum(alpha_dual) / dual
        D_q = D[x] * q_minus_e
        block = np.dot(self.tau(x).T, D_q)
        return block, np.sum(D_q[x == self.dim], axis=0)

    def max_entropy_dual(self, psi, many_classes):
        return psi / (self.many_values * many_classes)


class ExponentialFamilyStatistic(Statistic):
    """Feature following an exponential family f(x; theta) with sufficient statistic T(x).

    The family enters DSNGD only through the score in expectation coordinates
    g(x, theta*) = grad_{theta*} log f(x; theta*), see dual_score. Parameters are arrays whose last
    axis has length dim.
    """

    def transform(self, rv):
        return np.asarray(rv, dtype=float)[:, None]

    def dsngd_terms(self, x, dual, alpha_dual, q_minus_e):
        u_q = q_minus_e * (np.sum(alpha_dual) / alpha_dual)
        theta_star = (dual / alpha_dual).T
        g = self.dual_score(np.asarray(x, dtype=float), theta_star)
        block = np.einsum('ncd,nc->dc', g, u_q)
        w = 1. - np.einsum('cd,ncd->nc', theta_star, g)
        return block, np.sum(w * u_q, axis=0)

    def max_entropy_dual(self, psi, many_classes):
        return psi / many_classes * self.expectation(self.initial_natural())[:, None]

    def __repr__(self):
        arguments = ', '.join(f'{key}={value}' for key, value in vars(self).items() if key not in ('dim', 'dual_dim'))
        return f'{type(self).__name__}({arguments})'


class GaussianStatistic(ExponentialFamilyStatistic):
    """Gaussian with known variance, theta = mean / variance."""

    def __init__(self, variance=1.):
        super().__init__(1)
        self.variance = variance

    def log_partition(self, theta):
        return 0.5 * self.variance * theta[..., 0] ** 2

    def expectation(self, theta):
        return self.variance * theta

    def dual_score(self, x, theta_star):
        return ((x[:, None] - theta_star[None, :, 0]) / self.variance)[..., None]

    def initial_natural(self):
        return np.zeros(1)

    def random_natural(self, z):
        return z

    def sample(self, theta, rng):
        return rng.normal(self.variance * theta[:, 0], np.sqrt(self.variance))


class GaussianMeanVarianceStatistic(ExponentialFamilyStatistic):
    """Gaussian with unknown mean and variance, T(x) = (x, x^2) and theta = (mean / var, -1 / (2 var))."""

    def __init__(self):
        super().__init__(2)

    def transform(self, rv):
        rv = np.asarray(rv, dtype=float)
        return np.column_stack([rv, rv * rv])

    def log_partition(self, theta):
        return -theta[..., 0] ** 2 / (4. * theta[..., 1]) + 0.5 * np.log(-np.pi / theta[..., 1])

    def expectation(self, theta):
        mean, var = -theta[..., 0] / (2. * theta[..., 1]), -0.5 / theta[..., 1]
        return np.stack([mean, mean * mean + var], axis=-1)

    def dual_score(self, x, theta_star):
        mean = theta_star[None, :, 0]
        var = theta_star[None, :, 1] - mean * mean
        residual = x[:, None] - mean
        g_second = 0.5 * (residual * residual / var - 1.) / var
        g_mean = residual / var - 2. * mean * g_second
        return np.stack([g_mean, g_second], axis=-1)

    def initial_natural(self):
        return np.array([0., -0.5])

    def random_natural(self, z):
        mean, var = z[0], np.exp(z[1])
        return np.stack([mean / var, -0.5 / var])

    def sample(self, theta, rng):
        var = -0.5 / theta[:, 1]
        return rng.normal(var * theta[:, 0], np.sqrt(var))


class PoissonStatistic(ExponentialFamilyStatistic):
    """Poisson, theta = log(rate)."""

    def __init__(self):
        super().__init__(1)

    def log_partition(self, theta):
        return np.exp(theta[..., 0])

    def expectation(self, theta):
        return np.exp(theta)

    def dual_score(self, x, theta_star):
        return (x[:, None] / theta_star[None, :, 0] - 1.)[..., None]

    def initial_natural(self):
        return np.zeros(1)

    def random_natural(self, z):
        return z

    def sample(self, theta, rng):
        return rng.poisson(np.exp(theta[:, 0])).astype(float)


class ExponentialStatistic(ExponentialFamilyStatistic):
    """Exponential distribution, theta = -1 / mean < 0."""

    def __init__(self):
        super().__init__(1)

    def log_partition(self, theta):
        return -np.log(-theta[..., 0])

    def expectation(self, theta):
        return -1. / theta

    def dual_score(self, x, theta_star):
        mean = theta_star[None, :, 0]
        return (x[:, None] / (mean * mean) - 1. / mean)[..., None]

    def initial_natural(self):
        return -np.ones(1)

    def random_natural(self, z):
        return -np.exp(z)

    def sample(self, theta, rng):
        return rng.exponential(-1. / theta[:, 0])


class CanonicalFeatureStatistic(Statistic):
    """Naive Bayes feature vector. A feature is given by its number of values (categorical feature)
    or by an ExponentialFamilyStatistic (continuous or count feature)."""

    def __init__(self, features):
        self.canonical_statistics = [CanonicalStatistic(feature) if np.isscalar(feature) else feature
                                     for feature in features]
        self.m = features
        self.r = len(features)
        dim = np.sum([cs.dim for cs in self.canonical_statistics])
        self.discrete = all(cs.discrete for cs in self.canonical_statistics)
        self.psi = np.prod(self.m) if self.discrete else None
        self.continuous_dim = np.sum([cs.dim for cs in self.canonical_statistics if not cs.discrete])
        super().__init__(dim)
        self.dual_dim = np.sum([cs.dual_dim for cs in self.canonical_statistics])

    def transform(self, rvs):
        return np.hstack([cs.transform(rv) for cs, rv in zip(self.canonical_statistics, rvs.T)])

    def identity(self, rvs):
        return np.hstack([cs.identity(rv) for cs, rv in zip(self.canonical_statistics, rvs.T)])

    def slices(self):
        # Rows of each feature in the natural (beta) and in the dual parameters
        start, dual_start = 0, 0
        for cs in self.canonical_statistics:
            yield cs, slice(start, start + cs.dim), slice(dual_start, dual_start + cs.dual_dim)
            start, dual_start = start + cs.dim, dual_start + cs.dual_dim

    def log_partition(self, beta, discrete=True):
        # Sum of the feature log-partitions for each column (class) of beta
        log_partition = np.zeros(beta.shape[1])
        for cs, rows, _ in self.slices():
            if discrete or not cs.discrete:
                log_partition += cs.log_partition(beta[rows].T)
        return log_partition

    def expectation(self, beta):
        return np.vstack([cs.expectation(beta[rows].T).T for cs, rows, _ in self.slices()])

    def initial_natural(self):
        return np.concatenate([cs.initial_natural() for cs in self.canonical_statistics])

    def random_natural(self, z, continuous_scale=1.):
        beta = z.copy()
        for cs, rows, _ in self.slices():
            beta[rows] = cs.random_natural(z[rows]) if cs.discrete else cs.random_natural(continuous_scale * z[rows])
        return beta

    def sample(self, beta, rng):
        # One observation for each column of beta
        return np.column_stack([cs.sample(beta[rows].T, rng) for cs, rows, _ in self.slices()])



if __name__ == "__main__":
    T = CanonicalStatistic(3)
    x = np.array([0])
    print(T.tau(x).T)