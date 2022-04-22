import numpy as np
import itertools as iter
from tqdm import tqdm


class Algorithm(object):
    def __init__(self):
        self.tqdm = tqdm
        self.max_iterations_to_keep = 100
        # self.max_sample_to_adjust_lr = 300  # it is desirable to be close to max_iteations_to_keep
        self.single_learning_rate_parameter = False

        self.key = 'Empty'
        self.length = None

    def lr_training_function(self):
        return None

    def run_algorithm(self):
        return None

    def length_between_kept_iterations(self, n):
        m = min(self.max_iterations_to_keep, n)
        return n//m

    def set_run(self, many_iterations):
        # self.iter_per_epoch = sample_length // self.batch + bool(sample_length % self.batch)
        self.length = self.length_between_kept_iterations(many_iterations)

    def save_current_parameter(self, it):
        response = (it % self.length) == 0
        return response

    def adjust_lr_with_data(self, data, seed=0, progress_bar=True):
        run_algorithm_and_evaluate_in_f = self.lr_training_function(data)
        if self.single_learning_rate_parameter:
            gamma_exps = np.arange(-5, 2)
            gamma = np.power(10., gamma_exps)
            learning_rates = gamma
            best_lr = np.array([1.])
            best_err = run_algorithm_and_evaluate_in_f(best_lr)
        else:
            a_exps = np.arange(-4, 2)
            a = np.power(10., a_exps)
            b_exps = np.arange(-4, 2)
            b = np.power(10., b_exps)
            learning_rates = list(iter.product(a, b))
            best_lr = np.array([1., 1.])
            best_err = run_algorithm_and_evaluate_in_f(best_lr)
        if progress_bar:
            learning_rates = self.tqdm(learning_rates)
        for lr in learning_rates:
            try:
                # print("lr: ", lr)
                err = run_algorithm_and_evaluate_in_f(lr)
                # print("error: ", err)
                if err < best_err:
                    best_lr = lr
                    best_err = err
            except (OverflowError, np.linalg.linalg.LinAlgError, FloatingPointError):
                pass
        print("Best lr for", self.key, "is ", best_lr, "with error: ", best_err)
        # if best_lr[0] == 10. ** a_exps[0]:
        #     print("Decrease min a range")
        #     exit()
        # if (best_lr[0] == 10. ** a_exps[-1]):
        #     print("Increase max a range")
        #     exit()
        # if (best_lr[1] == 10. ** b_exps[0]):
        #     print("Decrease min b range")
        #     exit()
        # if (best_lr[1] == 10. ** b_exps[-1]):
        #     print("Increase max max  b range")
        #     exit()
        return best_lr

    @staticmethod
    def constant_learning_rate(i, learning_rate_param):
        return learning_rate_param

    @staticmethod
    def regular_learning_rate(i, learning_rate_param):
        a, b = learning_rate_param
        return a / (1. + b * i)

    @staticmethod
    def adagrad_learning_rate(gti, learning_rate_param):
        fudge_factor = 1e-8
        return learning_rate_param / np.sqrt(fudge_factor + gti)

    @staticmethod
    def adadelta_learning_rate(E_gti, E_updates):
        fudge_factor = 1e-8
        RMS_gti = np.sqrt(fudge_factor + E_gti)
        RMS_updates = np.sqrt(fudge_factor + E_updates)
        return RMS_updates / RMS_gti
