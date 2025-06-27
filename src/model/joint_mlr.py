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
        # self.trainer = Trainer()
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

    # def log_p(self, x, y):
    #     all_log_p = self.predict(x)
    #     n = len(y)
    #     y = y.copy().astype(int)
    #     log_p = all_log_p[:, y, range(n)]
    #     return log_p

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

        # pyx = self.compute_all_pyx(eta)
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

        param = (self.alpha.copy(), self.beta.copy())
        etas = optimizer.run(sample, starting_point=param, lr=best_lr, iter_keep=1000)
        best_err = np.sum(self.compute_metrics(true_model, etas[-5:]))

        for lr in tqdm(lrs, desc="Learning rate search..."):
            try:
                etas = optimizer.run(sample, starting_point=param, lr=lr, iter_keep=1000)
                err = np.sum(self.compute_metrics(true_model, etas[-5:]))
                # logging.info(f"Learning rate: {lr} with error: {err} ")

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
        kl = self.relative_entropy(true_pyx, log_true_model) - self.relative_entropy(true_pyx, log_pred)
        # kl = self.relative_entropy(true_pyx, log_pred)
        return -kl

    def compute_history_metrics(self, true_model):
        etas = self.history['etas']
        self.history['error'] = self.compute_metrics(true_model, etas)

    def relative_entropy(self, p, log_qs):
        kl = -p * log_qs  # + p * np.log(p)# + p[non_zeros] * np.log(p)[non_zeros]
        kl = np.sum(kl, axis=(1, 2))
        return kl

# s = 5

# m = np.array([3,5,10])
#model = JointMLR(s, m)
# model.set_random_eta(sigma=0.5)
# sample = JointMLRSampleIterator(model, 100, 2, 8)
# class Trainer:
#     def __init__(self):
#         return
#
#     def train(self, sample, algor: Algorithm, lr):
#
#     sample = data["sample"]
#     lr = data["lr"]
#
#     total_iter = data['total iterations']
#     self.set_run(total_iter)
#     parameters = []
#     current_parameter = data["start"].copy()
#     if progress_bar:
#         sample = self.tqdm(sample, total=total_iter)
#     i = 0
#     for obs in sample:
#         if self.save_current_parameter(i):
#             parameters.append(current_parameter.copy())
#         g = self.direction(obs, current_parameter)
#         r = self.learning_rate(i, lr)
#         current_parameter -= r * g
#         i = i + 1
#     parameters.append(current_parameter.copy())
#     return np.array(parameters)
#
# self.tqdm = tqdm
#         self.max_iterations_to_keep = 100
#         # self.max_sample_to_adjust_lr = 300  # it is desirable to be close to max_iteations_to_keep
#         self.single_learning_rate_parameter = False
#
#         self.key = 'Empty'
#         self.length = None
#
#     def length_between_kept_iterations(self, n):
#         m = min(self.max_iterations_to_keep, n)
#         return n//m
#
#     def set_run(self, many_iterations):
#         # self.iter_per_epoch = sample_length // self.batch + bool(sample_length % self.batch)
#         self.length = self.length_between_kept_iterations(many_iterations)
#
#     def save_current_parameter(self, it):
#         response = (it % self.length) == 0
#         return response
#
#     def adjust_lr_with_data(self, data, seed=0, progress_bar=True):
#         run_algorithm_and_evaluate_in_f = self.lr_training_function(data)
#         if self.single_learning_rate_parameter:
#             gamma_exps = np.arange(-5, 2)
#             gamma = np.power(10., gamma_exps)
#             learning_rates = gamma
#             best_lr = np.array([1.])
#             best_err = run_algorithm_and_evaluate_in_f(best_lr)
#         else:
#             a_exps = np.arange(-4, 2)
#             a = np.power(10., a_exps)
#             b_exps = np.arange(-4, 2)
#             b = np.power(10., b_exps)
#             learning_rates = list(iter.product(a, b))
#             best_lr = np.array([1., 1.])
#             best_err = run_algorithm_and_evaluate_in_f(best_lr)
#         if progress_bar:
#             learning_rates = self.tqdm(learning_rates)
#         for lr in learning_rates:
#             try:
#                 # print("lr: ", lr)
#                 err = run_algorithm_and_evaluate_in_f(lr)
#                 # print("error: ", err)
#                 if err < best_err:
#                     best_lr = lr
#                     best_err = err
#             except (OverflowError, np.linalg.linalg.LinAlgError, FloatingPointError):
#                 pass
#         print("Best lr for", self.key, "is ", best_lr, "with error: ", best_err)
#         # if best_lr[0] == 10. ** a_exps[0]:
#         #     print("Decrease min a range")
#         #     exit()
#         # if (best_lr[0] == 10. ** a_exps[-1]):
#         #     print("Increase max a range")
#         #     exit()
#         # if (best_lr[1] == 10. ** b_exps[0]):
#         #     print("Decrease min b range")
#         #     exit()
#         # if (best_lr[1] == 10. ** b_exps[-1]):
#         #     print("Increase max max  b range")
#         #     exit()
#         return best_lr
#
#     @staticmethod
#     def constant_learning_rate(i, learning_rate_param):
#         return learning_rate_param
#
#     @staticmethod
#     def regular_learning_rate(i, learning_rate_param):
#         a, b = learning_rate_param
#         return a / (1. + b * i)
#
#     @staticmethod
#     def adagrad_learning_rate(gti, learning_rate_param):
#         fudge_factor = 1e-8
#         return learning_rate_param / np.sqrt(fudge_factor + gti)
#
#     @staticmethod
#     def adadelta_learning_rate(E_gti, E_updates):
#         fudge_factor = 1e-8
#         RMS_gti = np.sqrt(fudge_factor + E_gti)
#         RMS_updates = np.sqrt(fudge_factor + E_updates)
#         return RMS_updates / RMS_gti
