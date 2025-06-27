import time
import itertools as it
import numpy as np


class Trainer:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def get_lr_range(self):
        if self.algorithm.single_learning_rate_parameter:
            gamma_exps = np.arange(-5, 2)
            gamma = np.power(10., gamma_exps)
            learning_rates = gamma
        else:
            a_exps = np.arange(-4, 2)
            a = np.power(10., a_exps)
            b_exps = np.arange(-4, 2)
            b = np.power(10., b_exps)
            learning_rates = list(it.product(a, b))
        return learning_rates

    def fit_lr(self, starting_parameter, sample):
        lrs = self.get_lr_range()
        for lr in lrs:
            self.fit()
        tools = self.algorithm.get_lr_ranges()
        sample_lr = sample
        return

    def fit(self, starting_param, sample, lr):
        tools = self.algorithm.get_tools()
        t = time.clock()
        history = self.algorithm.run(starting_param, sample, lr, **tools)
        history['time'] = time.clock() - t
        history['time/it'] = history['time'] / len(sample)
        return history
