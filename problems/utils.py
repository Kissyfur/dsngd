import numpy as np
import itertools


def numerical_gradient(f, a, epsilon=1e-6):
    dim = a.shape
    grad = np.zeros(dim)
    cartesian = [range(i) for i in dim]
    coordinates = itertools.product(*cartesian)
    for c in coordinates:
        h = np.zeros(dim)
        h[c] = epsilon
        d = numerical_derivative(f, a, h)
        grad[c] = d
    return grad


def numerical_derivative(f, a, h):
    h_length = np.sum(h)
    return (f(a + h) - f(a - h))/(2*h_length)


def cll(log_p, sample, param):
    e = evaluate_per_param(log_p, sample, param)
    return e


def evaluate_per_param(f, sample, params):
    n = len(sample)
    B = 100
    val = 0.
    for batch in range(0, n, B):
        s = sample[batch:batch + B]
        val += np.sum(f(s, params), axis=1)
    return val #/n


def ncll(log_p, sample, betas):
    n = len(sample)
    many_betas = len(betas)
    log_p_happened = np.ones(many_betas)
    obs_per_theta = n * 1. / (many_betas-1.)
    if not obs_per_theta.is_integer():
        print('obs_per_theta not integer in ncll')
        # exit()
    obs_per_theta = int(obs_per_theta)
    normalize = np.arange(0, n+1, obs_per_theta)
    normalize[0] = 1
    for i in range(1, many_betas):
        obs = sample[obs_per_theta * (i-1): obs_per_theta * i]
        # TX = sample[obs_per_theta * (i-1): obs_per_theta * i, 1:]
        lp = log_p(obs, betas[i:])
        log_p_happened[i:] += np.sum(lp, axis=1)
    log_p_happened /= normalize
    return log_p_happened


def kl_divergence(p, qs):
    # r = np.where(p != 0., p, [2.])
    # log_p = np.log(r)
    #
    # e = np.where(q != 0., q, [2.])
    # log_q = np.log(e)
    #
    # kl = -p * log_q + p * log_p  # is it better to... : real * (log_real -log_estim) ?
    # kl = np.where((q == 0) * (p != 0), -np.inf, kl)
    shape = qs.shape
    total_axis = len(shape)
    non_zeros = np.nonzero(p)
    transposed_axis = tuple(range(1, total_axis))+(0,)
    qsT = qs.transpose(transposed_axis)
    log_qsT = np.log(qsT[non_zeros])
    # log_qs = log_qsT.reshape(shape)
    # kl = -p[non_zeros] * np.log(qs.T[non_zeros]).T #+ p[non_zeros] * np.log(p)[non_zeros]
    kl = -p[non_zeros] * log_qsT.T # + p[non_zeros] * np.log(p)[non_zeros]
    kl = np.sum(kl, axis=1)
    return kl

