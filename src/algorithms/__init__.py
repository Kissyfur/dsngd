import numpy as np
import itertools as iter
from tqdm import tqdm


class LineSearch:
    def __init__(self, director_process, name='LS'):
        self.director_process = director_process
        self.name = name
        self.key = name
        self.tqdm = tqdm
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
            learning_rates = self.tqdm(learning_rates, desc=f"{self.key} learning-rate search")
        for lr in learning_rates:
            try:
                # print("lr: ", lr)
                err = run_algorithm_and_evaluate_in_f(lr)
                # print("error: ", err)
                if not np.isfinite(best_err) or err < best_err:
                    best_lr = lr
                    best_err = err
            except (OverflowError, np.linalg.LinAlgError, FloatingPointError):
                pass
        print("Best lr for", self.key, "is ", best_lr, "with error: ", best_err, flush=True)
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

    def lr_training_function(self, data):
        def run_algorithm_and_evaluate(lr):
            try:
                model = data["model_factory"]()
                optimizer = type(self)(model)
                sample = data["sample_factory"]()
                iter_keep = data.get("iter_keep", len(sample))
                etas = optimizer.run(sample, model.eta, lr=lr, iter_keep=iter_keep)
                score_tail = data.get("score_tail", 5)
                curve = data["validation_curve"](model, etas[-score_tail:])
                if not np.all(np.isfinite(curve)):
                    return np.inf
                return float(np.sum(curve[-score_tail:]))
            except (OverflowError, np.linalg.LinAlgError, FloatingPointError, ValueError):
                return np.inf

        return run_algorithm_and_evaluate

    @staticmethod
    def regular_learning_rate(i, learning_rate_param):
        a, b = learning_rate_param
        return a / (1. + b * i)
