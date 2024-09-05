import numpy as np
from algorithms.algorithm import Algorithm
import itertools as itools


class SGD(Algorithm):
    def __init__(self, gradient):
        super(SGD, self).__init__()
        self.key = 'SGD'
        self.direction = gradient
        self.run_algorithm = self.sgd
        self.lr_training_function = self.lr_cost_function
        self.learning_rate = self.regular_learning_rate

    def sgd(self, data, progress_bar=False):
        sample = data["sample"]
        lr = data["lr"]

        total_iter = data['total iterations']
        self.set_run(total_iter)
        parameters = []
        current_parameter = data["start"].copy()
        if progress_bar:
            sample = self.tqdm(sample, total=total_iter)
        i = 0
        for obs in sample:
            if self.save_current_parameter(i):
                parameters.append(current_parameter.copy())
            g = self.direction(obs, current_parameter)
            r = self.learning_rate(i, lr)
            current_parameter -= r * g
            i = i + 1
        parameters.append(current_parameter.copy())
        return np.array(parameters)

    def lr_cost_function(self, data):
        cost_function = data['cost_function']
        sample_creator = data['sample_lr_creator']

        def f(lr):
            sample = sample_creator()
            # sample = itools.islice(sample, 0, self.max_sample_to_adjust_lr)
            data_lr = {'sample': sample, 'start': data['start'], 'lr': lr,
                       'total iterations': data['total lr iterations']}
            params = self.run_algorithm(data_lr)
            err = cost_function(params[-10:])
            err = np.sum(err)
            return err
        return f


