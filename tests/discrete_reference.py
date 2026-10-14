"""Reference outputs of the discrete code path.

``compute()`` only uses the public API of the discrete implementation, so running it before and
after a change shows whether the discrete case is still reproduced. ``python tests/discrete_reference.py``
stores the outputs of the current code in ``tests/data/discrete_golden.npz``.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.algorithms.adaGrad import AdaGrad_JointMLR
from src.algorithms.dsngd import DSNGD_JointMLR
from src.algorithms.sgd import SGD_JointMLR
from src.algorithms.sngd import SNGD_JointMLR
from src.data.sample_creator import JointMLRSampleIterator
from src.model.joint_mlr import JointMLR

GOLDEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'discrete_golden.npz')

# Outputs that depend on to_dual or on the KL metric are compared with a tolerance; the rest exactly.
APPROXIMATE_KEYS = ('to_dual_alpha', 'to_dual_beta', 'sngd_alpha', 'sngd_beta', 'fit_SNGD_alpha', 'fit_SNGD_beta')
APPROXIMATE_SUFFIX = '_error'


def compute():
    out = {}

    problem = JointMLR(3, [2, 3])
    np.random.seed(5)
    problem.set_random_eta(0.4)
    out['eta_alpha'], out['eta_beta'] = problem.eta

    x = np.array([[0, 0], [1, 2], [0, 1], [1, 0], [1, 1], [0, 2]])
    y = np.array([0, 1, 2, 0, 1, 2])
    dsngd = DSNGD_JointMLR(JointMLR(3, [2, 3]))
    out['dsngd_alpha'], out['dsngd_beta'] = dsngd.aprox_natural_gradient_log_conditional_probability(
        (x, y), problem.eta, dsngd.max_entropy_dual_parameter())

    out['to_dual_alpha'], out['to_dual_beta'] = problem.to_dual(problem.eta)
    sngd = SNGD_JointMLR(JointMLR(3, [2, 3]))
    out['sngd_alpha'], out['sngd_beta'] = sngd.aprox_natural_gradient_log_conditional_probability(
        (x, y), problem.eta, problem.to_dual(problem.eta))

    out['max_entropy_alpha'], out['max_entropy_beta'] = DSNGD_JointMLR(
        JointMLR(4, [3, 5, 2])).max_entropy_dual_parameter()

    batches = list(JointMLRSampleIterator(problem, 40, epochs=2, batch=8, random_seed=3))
    out['sample_x'] = np.vstack([b[0] for b in batches])
    out['sample_y'] = np.concatenate([b[1] for b in batches])

    problem = JointMLR(10, [10, 5])
    np.random.seed(0)
    problem.set_random_eta(0.7)
    sample = JointMLRSampleIterator(problem, 25000, 1, 250, random_seed=0)
    sample_lr = JointMLRSampleIterator(problem, 2500, 1, 250, random_seed=1)
    for algo_class, lr in ((SGD_JointMLR, (0.1, 0.01)), (AdaGrad_JointMLR, 0.1),
                           (DSNGD_JointMLR, (0.001, 0.1)), (SNGD_JointMLR, (0.001, 0.1))):
        model = JointMLR(10, [10, 5])
        algo = algo_class(model)
        model.fit(sample, algo, lr=lr)
        model.compute_history_metrics(problem)
        name = algo_class.CLASS_NAME
        out[f'fit_{name}_alpha'], out[f'fit_{name}_beta'] = model.eta
        out[f'fit_{name}_error'] = model.history['error']
        if algo_class is not SNGD_JointMLR:
            out[f'best_lr_{name}'] = np.asarray(JointMLR(10, [10, 5]).find_best_lr(problem, sample_lr, algo),
                                                dtype=float)
    return out


if __name__ == '__main__':
    os.makedirs(os.path.dirname(GOLDEN_FILE), exist_ok=True)
    np.savez(GOLDEN_FILE, **compute())
    print('Saved', GOLDEN_FILE)
