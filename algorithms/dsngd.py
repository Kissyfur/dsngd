import numpy as np
from algorithms.algorithm import Algorithm
import itertools as itools


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class DSNGD(Algorithm):
    def __init__(self, natural_gradient_aproximation, dual_estimator, batch=1):
        super(DSNGD, self).__init__()
        self.key = 'DSNGD'
        self.direction = natural_gradient_aproximation
        self.dual_estimator = dual_estimator
        self.run_algorithm = self.dsngd
        self.lr_training_function = self.lr_cost_function
        self.batch = batch

    def dsngd(self, data, progress_bar=False):
        sample_big = data["sample_big"]
        sample = data["sample"]
        current_parameter = data["start"].copy()
        current_dual_parameter = data["dual_start"].copy()
        lr = data["lr"]
        total_iter = data['total iterations']
        self.set_run(total_iter)
        self.dual_estimator.batch = self.batch
        parameters = []
        if progress_bar:
            sample = self.tqdm(sample, total=total_iter)
        i = 0
        for obs in sample:
            obs_big = next(sample_big)
            if self.save_current_parameter(i):
                parameters.append(current_parameter.copy())
            g = self.direction(obs_big, obs, current_parameter, current_dual_parameter)
            r = self.regular_learning_rate(i, lr)
            current_parameter -= r * g
            data_dual = {'sample': [obs], 'start': current_dual_parameter, 'alpha': 1., 'total iterations': 1}
            current_dual_parameter = self.dual_estimator.run_algorithm(data_dual)[-1]
            i = i + 1
        parameters.append(current_parameter.copy())
        return np.array(parameters)

    def lr_cost_function(self, data):
        start = data['start']
        dual_start = data['dual_start']
        sample_creator = data['sample_lr_creator']
        sample_big_creator = data['sample_big_lr_creator']
        cost_function = data['cost_function']

        def f(lr):
            sample = sample_creator()
            # sample = itools.islice(sample, 0, self.max_sample_to_adjust_lr)

            sample_big = sample_big_creator()
            # sample_big = itools.islice(sample_big, 0, self.max_sample_to_adjust_lr)

            data_lr = {'sample_big': sample_big, 'sample': sample, 'start': start,
                       'dual_start': dual_start, 'lr': lr, 'total iterations': data['total lr iterations']}
            params = self.run_algorithm(data_lr)
            err = cost_function(params[-10:])
            err = np.sum(err)
            return err
        return f

