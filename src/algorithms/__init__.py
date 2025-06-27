import numpy as np
import itertools as iter


class LineSearch:
    def __init__(self, director_process, name='LS'):
        self.director_process = director_process
        self.name = name
        self.single_learning_rate_parameter = False
        self.lr_update = self.regular_learning_rate

    def run(self, sample, starting_point, lr, iter_keep, **kwargs):
        param = starting_point
        params = []
        length = len(sample) // iter_keep
        for it, obs in enumerate(sample):
            if it % length == 0:
                params.append(param.copy())
            d = self.director_process(obs, param)
            r = self.lr_update(it, lr)
            param -= r * d
        return params

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
    def regular_learning_rate(i, learning_rate_param):
        a, b = learning_rate_param
        return a / (1. + b * i)
