import copy

import numpy as np
from scipy.special import digamma, gammaln, logsumexp, polygamma


class Statistic:
    discrete = False
    input_dim = 1

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
        logits = np.concatenate([theta, np.zeros((1,) + theta.shape[1:])])
        return logsumexp(logits, axis=0)

    def expectation(self, theta):
        # Probabilities of all the values, the expectation of identity
        logits = np.concatenate([theta, np.zeros((1,) + theta.shape[1:])])
        return np.exp(logits - logsumexp(logits, axis=0))

    def initial_natural(self):
        return np.zeros(self.dim)

    def random_natural(self, z):
        return z

    def sample(self, theta, rng):
        cumulative = np.cumsum(self.expectation(theta), axis=0)
        values = np.sum(cumulative < rng.random(theta.shape[1]), axis=0)
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

    Parameters are laid out like beta: the components of the statistic along the first axis and one column per
    class (or observation). The family enters DSNGD only through H(theta*), the Hessian of the conjugate of the
    log-partition at the expectation parameter theta*, which dual_hessian_product applies to vectors in closed form.
    The score in expectation coordinates is g(x, theta*) = H(theta*) (T(x) - theta*).
    """

    def transform(self, rv):
        return np.asarray(rv, dtype=float)[:, None]

    def dual_score(self, x, theta_star):
        # g for every observation (axis 1) and every column of theta_star (axis 2)
        residual = self.transform(x).T[:, :, None] - theta_star[:, None, :]
        return self.dual_hessian_product(theta_star[:, None, :], residual)

    def dsngd_terms(self, x, dual, alpha_dual, q_minus_e):
        u_q = q_minus_e * (np.sum(alpha_dual) / alpha_dual)
        theta_star = dual / alpha_dual
        a = np.sum(u_q, axis=0)
        # g is affine in T(x), so the sum over the observations only needs T(x)^T u q
        block = self.dual_hessian_product(theta_star, np.dot(self.transform(x).T, u_q) - theta_star * a)
        # Sum over the observations of u * q * (1 - <theta*, g>), as theta* does not depend on them
        return block, a - np.sum(theta_star * block, axis=0)

    def max_entropy_dual(self, psi, many_classes):
        return psi / many_classes * self.expectation(self.initial_natural())[:, None]

    def __repr__(self):
        arguments = ', '.join(f'{key}={value}' for key, value in vars(self).items()
                              if key not in ('dim', 'dual_dim', 'input_dim') and not key.startswith('_'))
        return f'{type(self).__name__}({arguments})'


class GaussianStatistic(ExponentialFamilyStatistic):
    """Gaussian with known variance, theta = mean / variance."""

    def __init__(self, variance=1.):
        super().__init__(1)
        self.variance = variance

    def log_partition(self, theta):
        return 0.5 * self.variance * theta[0] ** 2

    def expectation(self, theta):
        return self.variance * theta

    def dual_hessian_product(self, theta_star, rho):
        return rho / self.variance

    def initial_natural(self):
        return np.zeros(1)

    def random_natural(self, z):
        return z

    def sample(self, theta, rng):
        return rng.normal(self.variance * theta[0], np.sqrt(self.variance))


class GaussianMeanVarianceStatistic(ExponentialFamilyStatistic):
    """Gaussian with unknown mean and variance, T(x) = (x, x^2) and theta = (mean / var, -1 / (2 var))."""

    def __init__(self):
        super().__init__(2)

    def transform(self, rv):
        rv = np.asarray(rv, dtype=float)
        return np.column_stack([rv, rv * rv])

    def log_partition(self, theta):
        return -theta[0] ** 2 / (4. * theta[1]) + 0.5 * np.log(-np.pi / theta[1])

    def expectation(self, theta):
        mean, var = -theta[0] / (2. * theta[1]), -0.5 / theta[1]
        return np.stack([mean, mean * mean + var])

    def dual_hessian_product(self, theta_star, rho):
        mean, var = theta_star[0], theta_star[1] - theta_star[0] ** 2
        return np.stack([((var + 2. * mean * mean) * rho[0] - mean * rho[1]) / var ** 2,
                         (rho[1] - 2. * mean * rho[0]) / (2. * var ** 2)])

    def initial_natural(self):
        return np.array([0., -0.5])

    def random_natural(self, z):
        mean, var = z[0], np.exp(z[1])
        return np.stack([mean / var, -0.5 / var])

    def sample(self, theta, rng):
        var = -0.5 / theta[1]
        return rng.normal(var * theta[0], np.sqrt(var))


class MultivariateGaussianStatistic(ExponentialFamilyStatistic):
    """Gaussian vector of dimension d with unknown mean and covariance.

    T(x) = (x, x_i x_j for i <= j) and theta = (P mean, -P_ii / 2, -P_ij for i < j), with P the inverse covariance.
    The expectation parameter is (mean, E[x_i x_j] for i <= j).
    """

    def __init__(self, d):
        super().__init__(d + d * (d + 1) // 2)
        self.d = d
        self.input_dim = d
        self._upper = np.triu_indices(d)

    def transform(self, rv):
        rv = np.asarray(rv, dtype=float).reshape(len(rv), self.d)
        return np.hstack([rv, rv[:, self._upper[0]] * rv[:, self._upper[1]]])

    def _symmetric(self, upper):
        # Symmetric matrices (..., d, d) from their upper triangles, given along the first axis
        upper = np.moveaxis(upper, 0, -1)
        matrix = np.zeros(upper.shape[:-1] + (self.d, self.d))
        matrix[..., self._upper[0], self._upper[1]] = upper
        matrix[..., self._upper[1], self._upper[0]] = upper
        return matrix

    def _natural(self, mean, precision):
        # Natural parameter, with its components along the first axis, of a mean and an inverse covariance
        quadratic = -(precision - precision * np.eye(self.d) / 2.)
        return np.concatenate([np.moveaxis((precision @ mean[..., None])[..., 0], -1, 0),
                               np.moveaxis(quadratic[..., self._upper[0], self._upper[1]], -1, 0)])

    def _moments(self, theta):
        quadratic = self._symmetric(theta[self.d:])
        covariance = np.linalg.inv(-(quadratic + quadratic * np.eye(self.d)))
        mean = (covariance @ np.moveaxis(theta[:self.d], 0, -1)[..., None])[..., 0]
        return mean, covariance

    def log_partition(self, theta):
        mean, covariance = self._moments(theta)
        return (0.5 * np.sum(np.moveaxis(theta[:self.d], 0, -1) * mean, axis=-1)
                + 0.5 * np.linalg.slogdet(2. * np.pi * covariance)[1])

    def expectation(self, theta):
        mean, covariance = self._moments(theta)
        second = covariance + mean[..., :, None] * mean[..., None, :]
        return np.concatenate([np.moveaxis(mean, -1, 0), np.moveaxis(second[..., self._upper[0], self._upper[1]], -1, 0)])

    def dual_hessian_product(self, theta_star, rho):
        # Derivative of the natural parameter in the direction rho of the expectation parameter
        theta_star, rho = np.broadcast_arrays(theta_star, rho)
        mean = np.moveaxis(theta_star[:self.d], 0, -1)
        precision = np.linalg.inv(self._symmetric(theta_star[self.d:]) - mean[..., :, None] * mean[..., None, :])
        rho_mean = np.moveaxis(rho[:self.d], 0, -1)
        d_covariance = (self._symmetric(rho[self.d:]) - rho_mean[..., :, None] * mean[..., None, :]
                        - mean[..., :, None] * rho_mean[..., None, :])
        d_precision = -precision @ d_covariance @ precision
        d_natural = self._natural(mean, d_precision)
        d_natural[:self.d] += np.moveaxis((precision @ rho_mean[..., None])[..., 0], -1, 0)
        return d_natural

    def initial_natural(self):
        return self._natural(np.zeros(self.d), np.eye(self.d))

    def random_natural(self, z):
        # Mean z_mean and covariance L L^T, with L lower triangular with diagonal exp(z)
        mean = np.moveaxis(z[:self.d], 0, -1)
        lower = np.tril(self._symmetric(z[self.d:]))
        diagonal = np.arange(self.d)
        lower[..., diagonal, diagonal] = np.exp(lower[..., diagonal, diagonal])
        return self._natural(mean, np.linalg.inv(lower @ np.swapaxes(lower, -1, -2)))

    def sample(self, theta, rng):
        mean, covariance = self._moments(theta)
        noise = rng.standard_normal(mean.shape)
        return mean + (np.linalg.cholesky(covariance) @ noise[..., None])[..., 0]


class PoissonStatistic(ExponentialFamilyStatistic):
    """Poisson, theta = log(rate)."""

    def __init__(self):
        super().__init__(1)

    def log_partition(self, theta):
        return np.exp(theta[0])

    def expectation(self, theta):
        return np.exp(theta)

    def dual_hessian_product(self, theta_star, rho):
        return rho / theta_star

    def initial_natural(self):
        return np.zeros(1)

    def random_natural(self, z):
        return z

    def sample(self, theta, rng):
        return rng.poisson(np.exp(theta[0])).astype(float)


class ExponentialStatistic(ExponentialFamilyStatistic):
    """Exponential distribution, theta = -1 / mean < 0."""

    def __init__(self):
        super().__init__(1)

    def log_partition(self, theta):
        return -np.log(-theta[0])

    def expectation(self, theta):
        return -1. / theta

    def dual_hessian_product(self, theta_star, rho):
        return rho / theta_star ** 2

    def initial_natural(self):
        return -np.ones(1)

    def random_natural(self, z):
        return -np.exp(z)

    def sample(self, theta, rng):
        return rng.exponential(-1. / theta[0])


class GammaStatistic(ExponentialFamilyStatistic):
    """Gamma with unknown shape k and rate l, T(x) = (x, log x) and theta = (-l, k - 1)."""

    def __init__(self, newton_steps=8):
        super().__init__(2)
        self.newton_steps = newton_steps

    def transform(self, rv):
        rv = np.asarray(rv, dtype=float)
        return np.column_stack([rv, np.log(rv)])

    def log_partition(self, theta):
        shape, rate = theta[1] + 1., -theta[0]
        return gammaln(shape) - shape * np.log(rate)

    def expectation(self, theta):
        shape, rate = theta[1] + 1., -theta[0]
        return np.stack([shape / rate, digamma(shape) - np.log(rate)])

    def shape_rate(self, theta_star):
        # Inverse of the expectation: Newton's method on log k - digamma(k) = log E[x] - E[log x], from the
        # approximation of Minka (2002)
        c = np.log(theta_star[0]) - theta_star[1]
        shape = (3. - c + np.sqrt((c - 3.) ** 2 + 24. * c)) / (12. * c)
        for _ in range(self.newton_steps):
            step = (np.log(shape) - digamma(shape) - c) / (1. / shape - polygamma(1, shape))
            shape = shape - step
            if np.max(np.abs(step / shape)) < 1e-12:
                break
        return shape, shape / theta_star[0]

    def dual_hessian_product(self, theta_star, rho):
        # (hess F)^-1 rho with hess F = [[k / l^2, 1 / l], [1 / l, trigamma(k)]]
        shape, rate = self.shape_rate(theta_star)
        trigamma = polygamma(1, shape)
        determinant = (shape * trigamma - 1.) / rate ** 2
        return np.stack([trigamma * rho[0] - rho[1] / rate, shape * rho[1] / rate ** 2 - rho[0] / rate]) / determinant

    def initial_natural(self):
        return np.array([-1., 0.])

    def random_natural(self, z):
        return np.stack([-np.exp(z[1]), np.exp(z[0]) - 1.])

    def sample(self, theta, rng):
        return rng.gamma(theta[1] + 1., -1. / theta[0])


def inverse_digamma(y, newton_steps=5):
    # Initialization of Minka (2000) followed by Newton's method
    x = np.where(y >= -2.22, np.exp(y) + 0.5, -1. / (y + 0.5772156649015329))
    for _ in range(newton_steps):
        x = x - (digamma(x) - y) / polygamma(1, x)
    return x


class DirichletStatistic(ExponentialFamilyStatistic):
    """Dirichlet on the simplex of R^K, T(x) = (log x_1, ..., log x_K) and theta = (a_1 - 1, ..., a_K - 1)."""

    def __init__(self, K, fixed_point_steps=10, newton_steps=15):
        super().__init__(K)
        self.K = K
        self.input_dim = K
        self.fixed_point_steps = fixed_point_steps
        self.newton_steps = newton_steps
        self._last = {}

    def transform(self, rv):
        return np.log(np.asarray(rv, dtype=float).reshape(len(rv), self.K))

    def log_partition(self, theta):
        a = theta + 1.
        return np.sum(gammaln(a), axis=0) - gammaln(np.sum(a, axis=0))

    def expectation(self, theta):
        a = theta + 1.
        return digamma(a) - digamma(np.sum(a, axis=0))

    def hessian_solve(self, a, rho):
        # (hess F)^-1 rho with hess F = diag(trigamma(a)) - trigamma(sum(a)) 1 1^T, by Sherman-Morrison
        d = polygamma(1, a)
        c = polygamma(1, np.sum(a, axis=0))
        z = rho / d
        return z + c * np.sum(z, axis=0) / (1. - c * np.sum(1. / d, axis=0)) / d

    def parameters(self, theta_star):
        # Inverse of the expectation: fixed point iteration of Minka (2000) followed by Newton's method on log(a),
        # with steps limited to a factor e. Newton starts from the previous solution when theta* barely changed, as
        # happens between consecutive DSNGD iterations.
        last = self._last.get(theta_star.shape)
        if last is not None and np.max(np.abs(last[0] - theta_star)) < 0.05:
            a = last[1]
        else:
            a = np.ones_like(theta_star)
            for _ in range(self.fixed_point_steps):
                a = inverse_digamma(digamma(np.sum(a, axis=0)) + theta_star)
        for _ in range(self.newton_steps):
            step = np.clip(self.hessian_solve(a, self.expectation(a - 1.) - theta_star) / a, -1., 1.)
            a = a * np.exp(-step)
            if np.max(np.abs(step)) < 1e-12:
                break
        self._last[theta_star.shape] = (theta_star.copy(), a)
        return a

    def dual_hessian_product(self, theta_star, rho):
        return self.hessian_solve(self.parameters(theta_star), rho)

    def initial_natural(self):
        return np.zeros(self.K)

    def random_natural(self, z):
        return np.exp(z) - 1.

    def sample(self, theta, rng):
        gamma = rng.gamma(theta + 1.)
        return (gamma / np.sum(gamma, axis=0)).T


class BetaStatistic(DirichletStatistic):
    """Beta with parameters (a, b), T(x) = (log x, log(1 - x)) and theta = (a - 1, b - 1)."""

    def __init__(self, fixed_point_steps=10, newton_steps=15):
        super().__init__(2, fixed_point_steps, newton_steps)
        self.input_dim = 1

    def transform(self, rv):
        rv = np.asarray(rv, dtype=float)
        return np.column_stack([np.log(rv), np.log1p(-rv)])

    def sample(self, theta, rng):
        return super().sample(theta, rng)[:, 0]


class CanonicalFeatureStatistic(Statistic):
    """Naive Bayes feature vector. A feature is given by its number of values (categorical feature)
    or by an ExponentialFamilyStatistic, which may take several columns of the data (multivariate feature)."""

    def __init__(self, features):
        self.canonical_statistics = []
        for feature in features:
            statistic = CanonicalStatistic(feature) if np.isscalar(feature) else feature
            # A statistic object repeated in the list is copied, so that each feature keeps its own state
            if any(statistic is other for other in self.canonical_statistics):
                statistic = copy.deepcopy(statistic)
            self.canonical_statistics.append(statistic)
        self.m = features
        self.r = len(features)
        dim = np.sum([cs.dim for cs in self.canonical_statistics])
        self.discrete = all(cs.discrete for cs in self.canonical_statistics)
        self.psi = np.prod(self.m) if self.discrete else None
        self.continuous_dim = np.sum([cs.dim for cs in self.canonical_statistics if not cs.discrete])
        super().__init__(dim)
        self.dual_dim = np.sum([cs.dual_dim for cs in self.canonical_statistics])

    def columns(self, rvs):
        # Values of each feature: a column of the data, or several for multivariate features
        start = 0
        for cs in self.canonical_statistics:
            yield rvs[:, start] if cs.input_dim == 1 else rvs[:, start:start + cs.input_dim]
            start += cs.input_dim

    def transform(self, rvs):
        return np.hstack([cs.transform(rv) for cs, rv in zip(self.canonical_statistics, self.columns(rvs))])

    def identity(self, rvs):
        return np.hstack([cs.identity(rv) for cs, rv in zip(self.canonical_statistics, self.columns(rvs))])

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
                log_partition += cs.log_partition(beta[rows])
        return log_partition

    def expectation(self, beta):
        return np.vstack([cs.expectation(beta[rows]) for cs, rows, _ in self.slices()])

    def initial_natural(self):
        return np.concatenate([cs.initial_natural() for cs in self.canonical_statistics])

    def random_natural(self, z, continuous_scale=1.):
        beta = z.copy()
        for cs, rows, _ in self.slices():
            beta[rows] = cs.random_natural(z[rows]) if cs.discrete else cs.random_natural(continuous_scale * z[rows])
        return beta

    def sample(self, beta, rng):
        # One observation for each column of beta
        return np.column_stack([cs.sample(beta[rows], rng) for cs, rows, _ in self.slices()])



if __name__ == "__main__":
    T = CanonicalStatistic(3)
    x = np.array([0])
    print(T.tau(x).T)
