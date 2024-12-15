import numpy as np
from algorithms.dsngd import DSNGD
import itertools as itools


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class SNGD(DSNGD):
    def __init__(self, natural_gradient_aproximation, to_dual, batch=1):
        super(SNGD, self).__init__(natural_gradient_aproximation, None, batch)
        self.key = 'SNGD'
        self.to_dual = to_dual
        self.run_algorithm = self.sngd

    def sngd(self, data, progress_bar=False):
        sample_big = data["sample_big"]
        sample = data["sample"]
        current_parameter = data["start"].copy()
        lr = data["lr"]
        total_iter = data['total iterations']
        self.set_run(total_iter)
        parameters = []
        if progress_bar:
            sample = self.tqdm(sample, total=total_iter)
        i = 0
        for obs, obs_big in zip(sample, sample_big):
            # obs_big = next(sample_big)
            if self.save_current_parameter(i):
                parameters.append(current_parameter.copy())
            current_dual_parameter = self.to_dual(current_parameter)
            g = self.direction(obs_big, obs, current_parameter, current_dual_parameter)
            r = self.regular_learning_rate(i, lr)
            current_parameter -= r * g
            i = i + 1
        parameters.append(current_parameter.copy())
        return np.array(parameters)

    def lr_cost_function(self, data):
        start = data['start']
        sample_creator = data['sample_lr_creator']
        sample_big_creator = data['sample_big_lr_creator']
        cost_function = data['cost_function']

        def f(lr):
            sample = sample_creator()
            # sample = itools.islice(sample, 0, self.max_sample_to_adjust_lr)

            sample_big = sample_big_creator()
            # sample_big = itools.islice(sample_big, 0, self.max_sample_to_adjust_lr)

            data_lr = {'sample_big': sample_big, 'sample': sample, 'start': start,
                    'lr': lr, 'total iterations': data['total lr iterations']}
            params = self.run_algorithm(data_lr)
            err = cost_function(params[-10:])
            err = np.sum(err)
            return err
        return f

