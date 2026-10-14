import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt

from src.model.joint_mlr import JointMLR
from src.model.statistic import (ExponentialStatistic, GaussianMeanVarianceStatistic, GaussianStatistic,
                                 PoissonStatistic)
from src.data.sample_creator import JointMLRSampleIterator
from tqdm import tqdm
from src.algorithms.sgd import SGD_JointMLR
from src.algorithms.dsngd import DSNGD_JointMLR
from src.grapher import plot_lines
from src.algorithms.sngd import SNGD_JointMLR
from src.algorithms.adaGrad import AdaGrad_JointMLR


logging.basicConfig(level=logging.INFO)

ALGORITHMS = {algo_class.CLASS_NAME: algo_class
              for algo_class in [SGD_JointMLR, AdaGrad_JointMLR, DSNGD_JointMLR, SNGD_JointMLR]}
FAMILIES = {'gaussian': GaussianStatistic, 'gaussian-mv': GaussianMeanVarianceStatistic,
            'poisson': PoissonStatistic, 'exponential': ExponentialStatistic}


def build_manifolds(features):
    M_1 = (10, [10, 5])
    M_2 = (20, [10, 5, 10, 5])
    M_3 = (30, [10, 5, 10, 5, 10, 5])
    manifolds = [M_1, M_2, M_3]
    if features == 'discrete':
        return manifolds
    if features == 'mixed':
        # Each 5-valued feature is replaced by Gaussian (mean and variance), Poisson and exponential features,
        # which keeps the dimension of the manifold
        continuous = lambda: [GaussianMeanVarianceStatistic(), PoissonStatistic(), ExponentialStatistic()]
        return [(s, sum([[v] if v == 10 else continuous() for v in m], [])) for s, m in manifolds]
    # Features of one family, as many as needed to match the dimension of the discrete manifold
    family = FAMILIES[features]
    return [(s, [family() for _ in range(int(np.ceil(sum(v - 1 for v in m) / family().dim)))])
            for s, m in manifolds]


def parse_args():
    parser = argparse.ArgumentParser(description='Compare the algorithms on the manifolds M1, M2 and M3.')
    parser.add_argument('--features', choices=['discrete', 'mixed'] + list(FAMILIES), default='discrete')
    parser.add_argument('--n', type=int, default=10000000, help='Length of the training sample')
    parser.add_argument('--n-lr', type=int, default=500 * 250, help='Length of the learning rate search sample')
    parser.add_argument('--experiments', type=int, default=1, help='Experiments per manifold and entropy')
    parser.add_argument('--support-size', type=int, default=100000,
                        help='Monte Carlo sample of x used to compute the KL divergence for continuous features')
    parser.add_argument('--algorithms', nargs='+', choices=list(ALGORITHMS), default=['SGD', 'AdaGrad', 'DSNGD'])
    parser.add_argument('--output', help='Path without extension where the figure and the curves are saved')
    parser.add_argument('--quiet', action='store_true', help='Hide the training progress bars')
    return parser.parse_args()


def save_curves(file_name, curves, lrs, labels, n):
    arrays = {f'curves_{i}_{j}': np.array([d[label] for label in labels]) for (i, j), d in curves.items()}
    np.savez(file_name + '.npz', labels=labels, n=n, **arrays,
             lrs=np.array([[i, j, exp_num, labels.index(name)] + list(np.atleast_1d(lr)) + [np.nan] * (2 - np.size(lr))
                           for (i, j, exp_num, name), lr in lrs.items()]))


if __name__ == "__main__":
    args = parse_args()
    n = args.n
    epochs = 1
    batch = 250
    n_lr = args.n_lr
    many_experiments = args.experiments
    entropies = [0.1, 0.7, 1]
    manifolds = build_manifolds(args.features)
    algorithm_classes = [ALGORITHMS[name] for name in args.algorithms]
    labels = [algo_class.CLASS_NAME for algo_class in algorithm_classes]
    curves = {}
    lrs = {}
    problems = {}
    for i, (s, m) in enumerate(manifolds):
        for j, sigma_sq in enumerate(entropies):
            d = {algo_class.CLASS_NAME: [] for algo_class in algorithm_classes}
            for exp_num in range(many_experiments):
                logging.info(f"Running experiment")
                logging.info(f"Features: {args.features}, Manifold: M{i + 1} {(s, m)},     Entropy: {sigma_sq},      "
                             f"Experiment num: {exp_num}")
                problem = JointMLR(s, m)
                np.random.seed(exp_num)
                problem.set_random_eta(sigma_sq)
                problem.set_pyx(args.support_size)
                problems[(i, j, exp_num)] = problem
                sample = JointMLRSampleIterator(problem, n, epochs, batch, random_seed=exp_num)
                sample_lr = JointMLRSampleIterator(problem, n_lr, 1, batch, random_seed=exp_num+1)
                for algo_class in algorithm_classes:
                    logging.info(f"Algorithm: {algo_class.CLASS_NAME}")
                    model = JointMLR(s, m)
                    algo = algo_class(model)
                    lr = model.find_best_lr(problem, sample_lr, algo)
                    model.fit(sample, algo, verbose=not args.quiet, lr=lr)
                    model.compute_history_metrics(problem)
                    d[algo.CLASS_NAME].append(model.history['error'])
                    lrs[(i, j, exp_num, algo.CLASS_NAME)] = lr
                    logging.info(f"{algo.CLASS_NAME} final error: {model.history['error'][-1]:.4e}")
            curves[(i, j)] = d
            if args.output:
                save_curves(args.output, curves, lrs, labels, n)

    # (manifold, entropy, algorithm, experiment, checkpoint)
    errors = np.array([[[curves[(i, j)][label] for label in labels] for j in range(3)] for i in range(3)])
    gr = np.median(errors, axis=3)
    up = np.percentile(errors, q=75, axis=3)
    dn = np.percentile(errors, q=25, axis=3)
    checkpoints = errors.shape[-1]
    x = np.arange(checkpoints) * n*epochs // checkpoints
    if args.output:
        plt.figure(figsize=(14, 9))
    plot_lines(x, gr, labels, y_labels=['M1', 'M2', 'M3'], x_labels=['High entropy', 'Medium entropy', 'Low entropy'],
               low_lines=up, high_lines=dn, file_name=args.output or '')

    print("finished")
