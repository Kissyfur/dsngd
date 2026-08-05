import numpy as np
import itertools
import time
import logging

from tqdm import tqdm
from src.algorithms import LineSearch
from scipy.special import logsumexp
from src.model.statistic import CanonicalStatistic, CanonicalFeatureStatistic

logging.basicConfig(level=logging.INFO)


class JointMLR:
    def __init__(self, s, m, name='model'):
        self.S = CanonicalStatistic(s)
        self.T = CanonicalFeatureStatistic(m)
        self.name = name
        self.alpha = np.zeros(self.S.dim)
        self.beta = np.zeros((self.T.dim, self.S.dim + 1))
        self.eta = (self.alpha, self.beta)
        self.history = {}
        self.pyx = None

    def set_random_eta(self, sigma):
        alpha = np.random.normal(0., sigma, size=self.S.dim)
        beta = np.random.normal(0., sigma, size=(self.T.dim, self.S.many_values))
        self.set_eta((alpha, beta))

    def set_eta(self, eta):
        if not self.valid_eta(eta):
            print("Can not set eta with unmatch size")
        self.alpha, self.beta = eta
        self.eta = (self.alpha, self.beta)

    def valid_eta(self, eta):
        if len(eta) != 2:
            return False
        if eta[0].shape != (self.S.dim,):
            return False
        if eta[1].shape != (self.T.dim, self.S.dim + 1):
            return False
        return True

    def log_measures(self, x, etas):
        tx = self.T.transform(x)
        alphas, betas = np.array([eta[0] for eta in etas]), np.array([eta[1] for eta in etas])

        alpha_s = np.hstack([alphas, np.zeros((len(alphas), 1))])
        log_measures = np.dot(tx, betas) + alpha_s
        return log_measures.transpose((1, 0, 2))

    def log_conditional_probabilities(self, x, etas):
        lm = self.log_measures(x, etas)
        ld = logsumexp(lm, axis=2)
        return (lm.T - ld.T).T

    def conditional_probabilities(self, x, etas):
        lcp = self.log_conditional_probabilities(x, etas)
        return np.exp(lcp)

    def compute_all_x(self):
        ranged_xd_values = [range(i) for i in self.T.m]
        return itertools.product(*ranged_xd_values)

    def compute_all_pyx(self, eta):
        x = np.array(list(self.compute_all_x()))
        log_numerators = self.log_measures(x, [eta])[0]
        log_denominator = logsumexp(log_numerators)
        log_pyx = log_numerators - log_denominator
        pyx = np.exp(log_pyx)
        return pyx

    def set_pyx(self):
        if self.pyx is not None:
            return
        self.pyx = self.compute_all_pyx(self.eta)

    def to_dual(self, eta):
        alpha, beta = eta
        alpha_dual, beta_dual = np.zeros(self.S.many_values), np.zeros((np.sum(self.T.m), self.S.many_values))

        x = np.array(list(self.compute_all_x()))
        log_numerators = self.log_measures(x, [eta])[0]
        log_denominator = logsumexp(log_numerators)
        log_pyx = log_numerators - log_denominator
        pyx = np.exp(log_pyx)

        tx_big = self.T.identity(x)

        alpha_dual[:] = np.sum(pyx, axis=0)
        beta_dual[:] = np.dot(tx_big.T, pyx)
        return alpha_dual, beta_dual

    def find_best_lr(self, true_model, sample, optimizer: LineSearch):
        exp = np.power(10., np.arange(-4, 2))
        lrs = list(itertools.product(exp, exp))
        best_lr = np.array([1., 1.])

        if optimizer.single_learning_rate_parameter:
            lrs = list(itertools.product(exp))
            best_lr = 1.

        etas = optimizer.run(sample, starting_point=(self.alpha.copy(), self.beta.copy()), lr=best_lr, iter_keep=100)
        best_err = np.sum(self.compute_metrics(true_model, etas[-5:]))

        for lr in tqdm(lrs, desc="Learning rate search..."):
            try:
                etas = optimizer.run(sample, starting_point=(self.alpha.copy(), self.beta.copy()), lr=lr, iter_keep=100)
                err = np.sum(self.compute_metrics(true_model, etas[-5:]))

                if err < best_err:
                    best_lr = lr
                    best_err = err
            except (OverflowError, np.linalg.LinAlgError, FloatingPointError):
                pass
        logging.info(f"Best learning rate: {best_lr} with error: {best_err} ")
        return best_lr

    def fit(self, sample, optimizer: LineSearch, lr, verbose=False, iter_keep=100):
        n = len(sample)
        history = {}
        param = (self.alpha.copy(), self.beta.copy())
        t = time.time()
        history['etas'] = optimizer.run(sample, starting_point=param, lr=lr, verbose=verbose, iter_keep=iter_keep)
        history["time"] = time.time() - t
        history["time/it"] = history["time"] / n
        self.set_eta(history['etas'][-1])
        self.history = history

    def compute_metrics(self, true_model, estimations):
        true_model.set_pyx()
        true_pyx = true_model.pyx
        all_x = np.array(list(self.compute_all_x()))
        log_pred = self.log_conditional_probabilities(all_x, estimations)
        log_true_model = self.log_conditional_probabilities(all_x, [true_model.eta])
        kl = self.relative_entropy(true_pyx, log_pred) - self.relative_entropy(true_pyx, log_true_model)
        return kl

    def compute_history_metrics(self, true_model):
        etas = self.history['etas']
        self.history['error'] = self.compute_metrics(true_model, etas)

    def relative_entropy(self, p, log_qs):
        kl = -p * log_qs
        kl = np.sum(kl, axis=(1, 2))
        return kl
