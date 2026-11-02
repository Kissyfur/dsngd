import copy

import numpy as np
from scipy.special import digamma, gammaln, i0e, i1e, logsumexp, zeta


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

    def auxiliary_point(self, dual, alpha_dual, previous=None):
        # Auxiliary point of DSNGD for the features of a group, with the components along the first axis and the
        # features along the second, from their rows of the dual parameter: theta*, the average of the statistic.
        # Families that keep the point in other coordinates override it, and previous is their point at the previous
        # iteration.
        return dual[:self.dim] / alpha_dual

    def dsngd_block(self, theta_star, m, a):
        # Theorem 3.2 for the features of a group, summed over the mini-batch (Remark 4.12): the block
        # H(theta*) (M - a theta*) of the direction, and its inner products with theta* summed over the features. m
        # holds the weighted sums of identity, whose first dim components belong to the statistic. Families whose
        # product by H simplifies override it.
        block = self.dual_hessian_product(theta_star, m[:self.dim] - theta_star * a)
        return block, np.sum(theta_star * block, axis=(0, 1))


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
        values = np.sum(cumulative < rng.random(theta.shape[1:]), axis=0)
        return np.minimum(values, self.dim)

    def dual_hessian_product(self, theta_star, rho):
        # theta* = (p_1, ..., p_{m-1}) and H(theta*) rho = rho_i / p_i + sum(rho) / p_m
        return rho / theta_star + np.sum(rho, axis=0) / (1. - np.sum(theta_star, axis=0))

    def auxiliary_point(self, dual, alpha_dual, previous=None):
        # The probabilities of all the values
        return dual / alpha_dual

    def dsngd_block(self, probabilities, m, a):
        # Proposition 3.3 summed over the mini-batch. With N_v the weighted counts of the values and p_v their
        # probabilities, H(theta*) (M - a theta*) = N_i / p_i - N_m / p_m, whose inner product with theta* is
        # a - N_m / p_m, so no product by H is needed
        ratios = m / probabilities
        return ratios[:-1] - ratios[-1], np.sum(a - ratios[-1], axis=0)

    def max_entropy_dual(self, psi, many_classes):
        return psi / (self.many_values * many_classes)

    def __repr__(self):
        return f'{type(self).__name__}({self.many_values})'


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

    def max_entropy_dual(self, psi, many_classes):
        return psi / many_classes * self.expectation(self.initial_natural())[:, None]

    def __repr__(self):
        arguments = ', '.join(f'{key}={value}' for key, value in vars(self).items()
                              if key not in ('dim', 'dual_dim', 'input_dim') and not key.startswith('_'))
        return f'{type(self).__name__}({arguments})'


class NaturalPointStatistic(ExponentialFamilyStatistic):
    """Family whose natural parameter theta is implicit in the expectation parameter theta*, while theta* and H(theta*)
    are explicit in theta, as Gamma, Beta, Dirichlet and von Mises. natural finds theta from theta* with Newton's
    method, natural_point gives theta* and the factors of H at theta, and hessian_product applies H with them.

    DSNGD keeps the auxiliary point of these families in natural coordinates. The first point is the solution of
    Newton's method at the average of the statistic, and then the point follows the average with one Newton step per
    iteration (newton_step), which uses the H of the point.
    """

    def dual_hessian_product(self, theta_star, rho):
        return self.hessian_product(self.natural_point(self.natural(theta_star))[1], rho)

    def auxiliary_point(self, dual, alpha_dual, previous=None):
        average = dual / alpha_dual
        return self.natural_point(self.natural(average) if previous is None else self.newton_step(previous, average))

    def dsngd_block(self, point, m, a):
        theta_star, factors = point
        block = self.hessian_product(factors, m - theta_star * a)
        return block, np.sum(theta_star * block, axis=(0, 1))


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
        # The draws of rng.normal(mean, standard deviation), which is slower with arrays of means
        return self.variance * theta[0] + np.sqrt(self.variance) * rng.standard_normal(theta[0].shape)


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
        # The draws of rng.normal(mean, standard deviation), which is slower with arrays of parameters
        var = -0.5 / theta[1]
        return var * theta[0] + np.sqrt(var) * rng.standard_normal(var.shape)


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

    def _inverse(self, matrices):
        # For d = 1 the inverse is the reciprocal, as computed by LAPACK, without its cost for each matrix
        return 1. / matrices if self.d == 1 else np.linalg.inv(matrices)

    def _cholesky(self, matrices):
        return np.sqrt(matrices) if self.d == 1 else np.linalg.cholesky(matrices)

    def _moments(self, theta):
        quadratic = self._symmetric(theta[self.d:])
        covariance = self._inverse(-(quadratic + quadratic * np.eye(self.d)))
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
        precision = self._inverse(self._symmetric(theta_star[self.d:]) - mean[..., :, None] * mean[..., None, :])
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
        return self._natural(mean, self._inverse(lower @ np.swapaxes(lower, -1, -2)))

    def sample(self, theta, rng):
        mean, covariance = self._moments(theta)
        noise = rng.standard_normal(mean.shape)
        return mean + (self._cholesky(covariance) @ noise[..., None])[..., 0]

    def sample_classes(self, theta, classes, rng):
        # One observation for each entry of classes, with the parameters of its class along the last axis of theta.
        # The moments and their Cholesky factors are computed once for each class.
        mean, covariance = self._moments(theta)
        mean, factor = mean[..., classes, :], self._cholesky(covariance)[..., classes, :, :]
        noise = rng.standard_normal(mean.shape)
        return mean + (factor @ noise[..., None])[..., 0]


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


class VonMisesStatistic(NaturalPointStatistic):
    """Von Mises distribution of an angle (in radians), T(x) = (cos x, sin x) and theta = kappa (cos mu, sin mu)."""

    def __init__(self, newton_steps=10):
        super().__init__(2)
        self.newton_steps = newton_steps

    def transform(self, rv):
        rv = np.asarray(rv, dtype=float)
        return np.column_stack([np.cos(rv), np.sin(rv)])

    @staticmethod
    def mean_resultant(kappa):
        # A(kappa) = I_1(kappa) / I_0(kappa), the length of the expectation parameter
        return i1e(kappa) / i0e(kappa)

    def log_partition(self, theta):
        kappa = np.sqrt(np.sum(theta * theta, axis=0))
        return np.log(2. * np.pi) + np.log(i0e(kappa)) + kappa

    def expectation(self, theta):
        kappa = np.sqrt(np.sum(theta * theta, axis=0))
        safe = np.maximum(kappa, 1e-12)
        return theta * np.where(kappa > 1e-12, self.mean_resultant(safe) / safe, 0.5)

    def concentration(self, length):
        # Inverse of A by Newton's method, from the approximation of Banerjee et al. (2005)
        kappa = length * (2. - length * length) / (1. - length * length)
        for _ in range(self.newton_steps):
            a = self.mean_resultant(kappa)
            step = (a - length) / (1. - a / kappa - a * a)
            kappa = np.maximum(kappa - step, 1e-12)
            if np.max(np.abs(step / kappa)) < 1e-12:
                break
        return kappa

    def natural(self, theta_star):
        length = np.maximum(np.sqrt(np.sum(theta_star * theta_star, axis=0)), 1e-12)
        return self.concentration(length) * theta_star / length

    def natural_point(self, theta):
        # theta* and the factors of H: kappa, A(kappa), A'(kappa) = 1 - A / kappa - A^2 and the mean direction u. At the
        # uniform distribution, theta = 0, u is 0 and H = 2 I.
        kappa = np.maximum(np.sqrt(np.sum(theta * theta, axis=0)), 1e-12)
        a = self.mean_resultant(kappa)
        u = theta / kappa
        return a * u, (kappa, a, 1. - a / kappa - a * a, u)

    @staticmethod
    def hessian_product(factors, rho):
        # (hess F)^-1 rho with hess F = A'(kappa) u u^T + A(kappa) / kappa (I - u u^T)
        kappa, a, derivative, u = factors
        radial = np.sum(u * rho, axis=0)
        return radial / derivative * u + kappa / a * (rho - radial * u)

    def newton_step(self, point, average):
        # The direction of the average, and Newton's step on kappa toward A(kappa) = |average|, with decreases limited to
        # a factor e
        _, (kappa, a, derivative, _) = point
        length = np.maximum(np.sqrt(np.sum(average * average, axis=0)), 1e-12)
        return np.maximum(kappa + (length - a) / derivative, kappa / np.e) * average / length

    def initial_natural(self):
        return np.zeros(2)

    def random_natural(self, z):
        return z

    def sample(self, theta, rng):
        return rng.vonmises(np.arctan2(theta[1], theta[0]), np.sqrt(np.sum(theta * theta, axis=0)))


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
        # The draws of rng.exponential(mean), which is slower with arrays of means
        return -1. / theta[0] * rng.standard_exponential(theta[0].shape)


class GammaStatistic(NaturalPointStatistic):
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
            step = (np.log(shape) - digamma(shape) - c) / (1. / shape - trigamma(shape))
            shape = shape - step
            if np.max(np.abs(step / shape)) < 1e-12:
                break
        return shape, shape / theta_star[0]

    def natural(self, theta_star):
        shape, rate = self.shape_rate(theta_star)
        return np.stack([-rate, shape - 1.])

    def natural_point(self, theta):
        # theta* and the factors of H: the shape, the rate and trigamma(shape)
        shape, rate = theta[1] + 1., -theta[0]
        return self.expectation(theta), (shape, rate, trigamma(shape))

    @staticmethod
    def hessian_product(factors, rho):
        # (hess F)^-1 rho with hess F = [[k / l^2, 1 / l], [1 / l, trigamma(k)]]
        shape, rate, trigamma_shape = factors
        determinant = (shape * trigamma_shape - 1.) / rate ** 2
        return np.stack([trigamma_shape * rho[0] - rho[1] / rate,
                         shape * rho[1] / rate ** 2 - rho[0] / rate]) / determinant

    def newton_step(self, point, average):
        # Newton's step H(theta*) (average - theta*) on the logarithms of the shape and the rate, limited to a factor e
        theta_star, factors = point
        shape, rate, _ = factors
        step = self.hessian_product(factors, average - theta_star)
        return np.stack([-rate * np.exp(np.clip(-step[0] / rate, -1., 1.)),
                         shape * np.exp(np.clip(step[1] / shape, -1., 1.)) - 1.])

    def initial_natural(self):
        return np.array([-1., 0.])

    def random_natural(self, z):
        return np.stack([-np.exp(z[1]), np.exp(z[0]) - 1.])

    def sample(self, theta, rng):
        return rng.gamma(theta[1] + 1., -1. / theta[0])


def trigamma(x):
    # polygamma(1, x), which scipy computes as zeta(2, x) after also computing digamma(x) for the case n = 0
    return zeta(2., x)


def inverse_digamma(y, newton_steps=5):
    # Initialization of Minka (2000) followed by Newton's method. Both branches are evaluated, and the unused one
    # divides by zero at y = -0.5772...
    with np.errstate(divide='ignore'):
        x = np.where(y >= -2.22, np.exp(y) + 0.5, -1. / (y + 0.5772156649015329))
    for _ in range(newton_steps):
        x = x - (digamma(x) - y) / trigamma(x)
    return x


class DirichletStatistic(NaturalPointStatistic):
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

    def natural(self, theta_star):
        return self.parameters(theta_star) - 1.

    def natural_point(self, theta):
        # theta* and the factors of H: a, trigamma(a) and trigamma(sum(a))
        a = theta + 1.
        return self.expectation(theta), (a, trigamma(a), trigamma(np.sum(a, axis=0)))

    @staticmethod
    def hessian_product(factors, rho):
        # (hess F)^-1 rho with hess F = diag(trigamma(a)) - trigamma(sum(a)) 1 1^T, by Sherman-Morrison
        _, d, c = factors
        z = rho / d
        return z + c * np.sum(z, axis=0) / (1. - c * np.sum(1. / d, axis=0)) / d

    def log_step(self, point, average):
        # Newton's step H(theta*) (average - theta*) on log(a), limited to a factor e
        theta_star, factors = point
        return np.clip(self.hessian_product(factors, average - theta_star) / factors[0], -1., 1.)

    def newton_step(self, point, average):
        return point[1][0] * np.exp(self.log_step(point, average)) - 1.

    def parameters(self, theta_star):
        # Inverse of the expectation: fixed point iteration of Minka (2000) followed by Newton's method on log(a).
        # Newton starts from the previous solution when theta* barely changed.
        last = self._last.get(theta_star.shape)
        if last is not None and np.max(np.abs(last[0] - theta_star)) < 0.05:
            a = last[1]
        else:
            a = np.ones_like(theta_star)
            for _ in range(self.fixed_point_steps):
                a = inverse_digamma(digamma(np.sum(a, axis=0)) + theta_star)
        for _ in range(self.newton_steps):
            step = self.log_step(self.natural_point(a - 1.), theta_star)
            a = a * np.exp(step)
            if np.max(np.abs(step)) < 1e-12:
                break
        self._last[theta_star.shape] = (theta_star.copy(), a)
        return a

    def initial_natural(self):
        return np.zeros(self.K)

    def random_natural(self, z):
        return np.exp(z) - 1.

    def sample(self, theta, rng):
        # The draws of rng.gamma(theta + 1), whose unit scale makes them standard gamma draws
        gamma = rng.standard_gamma(theta + 1.)
        return np.moveaxis(gamma / np.sum(gamma, axis=0), 0, -1)


class BetaStatistic(DirichletStatistic):
    """Beta with parameters (a, b), T(x) = (log x, log(1 - x)) and theta = (a - 1, b - 1)."""

    def __init__(self, fixed_point_steps=10, newton_steps=15):
        super().__init__(2, fixed_point_steps, newton_steps)
        self.input_dim = 1

    def transform(self, rv):
        rv = np.asarray(rv, dtype=float)
        return np.column_stack([np.log(rv), np.log1p(-rv)])

    def sample(self, theta, rng):
        return super().sample(theta, rng)[..., 0]


class FeatureGroup:
    """Features with the same family and parameters, which are computed together.

    Feature after feature, it holds their columns of the data and their rows of beta and of the dual parameter.
    """

    def __init__(self, statistic):
        self.statistic = statistic
        self.size = 0
        self.columns, self.rows, self.dual_rows = [], [], []

    def add(self, column, row, dual_row):
        cs = self.statistic
        self.size += 1
        self.columns = np.append(self.columns, np.arange(column, column + cs.input_dim)).astype(int)
        self.rows = np.append(self.rows, np.arange(row, row + cs.dim)).astype(int)
        self.dual_rows = np.append(self.dual_rows, np.arange(dual_row, dual_row + cs.dual_dim)).astype(int)

    def values(self, rvs):
        # One row for each observation and feature
        values = rvs[:, self.columns].reshape(len(rvs) * self.size, self.statistic.input_dim)
        return values[:, 0] if self.statistic.input_dim == 1 else values

    def stack(self, parameters):
        # From rows of the group (of beta or of the dual parameter) to (component, feature, ...), the layout of the
        # parameters of a statistic
        return np.swapaxes(parameters.reshape((self.size, -1) + parameters.shape[1:]), 0, 1)

    def unstack(self, parameters):
        return np.swapaxes(parameters, 0, 1).reshape((-1,) + parameters.shape[2:])


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
        self.input_columns = sum(cs.input_dim for cs in self.canonical_statistics)
        # Features of the same family and parameters are computed together
        groups, column = {}, 0
        for cs, rows, dual_rows in self.slices():
            groups.setdefault(repr(cs), FeatureGroup(cs)).add(column, rows.start, dual_rows.start)
            column += cs.input_dim
        self.groups = list(groups.values())
        # Columns of identity that form the statistic
        self.expectation_rows = np.concatenate([dual_rows.start + np.arange(cs.dim)
                                                for cs, _, dual_rows in self.slices()])

    def transform(self, rvs):
        tx = np.empty((len(rvs), self.dim))
        for group in self.groups:
            tx[:, group.rows] = group.statistic.transform(group.values(rvs)).reshape(len(rvs), -1)
        return tx

    def identity(self, rvs):
        ix = np.empty((len(rvs), self.dual_dim))
        for group in self.groups:
            ix[:, group.dual_rows] = group.statistic.identity(group.values(rvs)).reshape(len(rvs), -1)
        return ix

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

    def sample(self, beta, rng, classes=None):
        # One observation for each column of beta or, given classes, for each class, a column of beta. Statistics with
        # sample_classes compute what they need once for each class.
        observations = beta if classes is None else beta[:, classes]
        x = np.empty((observations.shape[1], self.input_columns))
        for group in self.groups:
            if classes is not None and hasattr(group.statistic, 'sample_classes'):
                values = group.statistic.sample_classes(group.stack(beta[group.rows]), classes, rng)
            else:
                values = group.statistic.sample(group.stack(observations[group.rows]), rng)
            x[:, group.columns] = np.moveaxis(values, 0, 1).reshape(len(x), -1)
        return x



if __name__ == "__main__":
    T = CanonicalStatistic(3)
    x = np.array([0])
    print(T.tau(x).T)
