import argparse
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from src.model.joint_mlr import JointMLR
from src.model.statistic import (BetaStatistic, DirichletStatistic, ExponentialStatistic, GammaStatistic,
                                 GaussianMeanVarianceStatistic, GaussianStatistic, MultivariateGaussianStatistic,
                                 PoissonStatistic)
from src.data.sample_creator import JointMLRSampleIterator
from tqdm import tqdm
from src.algorithms.sgd import SGD_JointMLR
from src.algorithms.dsngd import DSNGD_JointMLR
from src.grapher import plot_lines
from src.algorithms.sngd import SNGD_JointMLR
from src.algorithms.adaGrad import AdaGrad_JointMLR
from src.progress import Progress


logging.basicConfig(level=logging.INFO)

ALGORITHMS = {algo_class.CLASS_NAME: algo_class
              for algo_class in [SGD_JointMLR, AdaGrad_JointMLR, DSNGD_JointMLR, SNGD_JointMLR]}
FAMILIES = {'gaussian': GaussianStatistic, 'gaussian-mv': GaussianMeanVarianceStatistic,
            'poisson': PoissonStatistic, 'exponential': ExponentialStatistic, 'gamma': GammaStatistic,
            'beta': BetaStatistic}
# Features replacing each pair of categorical features with 10 and 5 values (statistic of dimension 13)
BLOCKS = {
    'mixed': lambda: [10, GaussianMeanVarianceStatistic(), PoissonStatistic(), ExponentialStatistic()],
    'multivariate-gaussian': lambda: [MultivariateGaussianStatistic(3), MultivariateGaussianStatistic(1),
                                      MultivariateGaussianStatistic(1)],
    'dirichlet': lambda: [DirichletStatistic(5), DirichletStatistic(4), DirichletStatistic(4)],
}
ENTROPIES = ['high', 'medium', 'low']


def build_manifolds(features):
    M_1 = (10, [10, 5])
    M_2 = (20, [10, 5, 10, 5])
    M_3 = (30, [10, 5, 10, 5, 10, 5])
    manifolds = [M_1, M_2, M_3]
    if features == 'discrete':
        return manifolds
    if features in BLOCKS:
        # Each pair of categorical features is replaced by a block of features, which keeps the dimension of the
        # manifold. The mixed block keeps the 10-valued feature and replaces the 5-valued one by Gaussian (mean and
        # variance), Poisson and exponential features.
        return [(s, sum([BLOCKS[features]() for _ in range(len(m) // 2)], [])) for s, m in manifolds]
    # Features of one family, as many as needed to match the dimension of the discrete manifold
    family = FAMILIES[features]
    return [(s, [family() for _ in range(int(np.ceil(sum(v - 1 for v in m) / family().dim)))])
            for s, m in manifolds]


def parse_args():
    parser = argparse.ArgumentParser(description='Compare the algorithms on the manifolds M1, M2 and M3.')
    parser.add_argument('--features', choices=['discrete'] + list(FAMILIES) + list(BLOCKS), default='discrete')
    parser.add_argument('--n', type=int, default=10000000, help='Length of the training sample')
    parser.add_argument('--n-lr', type=int, default=2000 * 250, help='Length of the learning rate search sample')
    parser.add_argument('--experiments', type=int, default=1, help='Experiments per manifold and entropy')
    parser.add_argument('--support-size', type=int, default=100000,
                        help='Monte Carlo sample of x used to compute the KL divergence for continuous features')
    parser.add_argument('--algorithms', nargs='+', choices=list(ALGORITHMS), default=['SGD', 'AdaGrad', 'DSNGD'])
    parser.add_argument('--manifolds', nargs='+', type=int, choices=[1, 2, 3], default=[1, 2, 3],
                        help='Run only these manifolds (M1, M2, M3)')
    parser.add_argument('--entropies', nargs='+', choices=ENTROPIES, default=ENTROPIES,
                        help='Run only these entropies')
    parser.add_argument('--output', help='Path without extension where the figure and the curves are saved. The '
                                         'manifolds and entropies run replace theirs in the curves of a previous run, '
                                         'and the others are kept')
    parser.add_argument('--quiet', action='store_true', help='Hide the training progress bars')
    return parser.parse_args()


def check_previous(file_name, labels, n, many_experiments):
    # The curves of a previous run, which this one completes or updates, must have the same algorithms, length of the
    # training sample and experiments per manifold and entropy
    if not os.path.exists(file_name + '.npz'):
        return
    with np.load(file_name + '.npz') as previous:
        stored = [str(label) for label in previous['labels']]
        experiments = {previous[key].shape[1] for key in previous.files if key.startswith('curves_')}
        if stored != labels or int(previous['n']) != n or experiments != {many_experiments}:
            sys.exit(f"{file_name}.npz holds {'/'.join(map(str, sorted(experiments)))} experiments of "
                     f"{', '.join(stored)} on {int(previous['n'])} observations; run with these settings to replace "
                     f"some of its manifolds and entropies, or choose another --output")


def save_curves(file_name, curves, lrs, times, labels, n):
    arrays = {f'curves_{i}_{j}': np.array([d[label] for label in labels]) for (i, j), d in curves.items()}
    lr_rows = [[i, j, exp_num, labels.index(name)] + list(np.atleast_1d(lr)) + [np.nan] * (2 - np.size(lr))
               for (i, j, exp_num, name), lr in lrs.items()]
    # manifold, entropy, experiment, algorithm, training seconds, microseconds per sample
    time_rows = [[i, j, exp_num, labels.index(name), t, 1e6 * t / n] for (i, j, exp_num, name), t in times.items()]
    if os.path.exists(file_name + '.npz'):
        # The manifolds and entropies of previous runs that this one has not run again are kept
        with np.load(file_name + '.npz') as previous:
            arrays = {**{key: previous[key] for key in previous.files if key.startswith('curves_')}, **arrays}
            lr_rows = [list(row) for row in previous['lrs'] if (int(row[0]), int(row[1])) not in curves] + lr_rows
            time_rows = [list(row) for row in previous['times'] if (int(row[0]), int(row[1])) not in curves] + time_rows
    np.savez(file_name + '.npz', labels=labels, n=n, **arrays, lrs=np.array(lr_rows), times=np.array(time_rows))


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
    if args.output:
        check_previous(args.output, labels, n, many_experiments)
    manifold_indices = sorted(set(m - 1 for m in args.manifolds))
    entropy_indices = sorted(set(ENTROPIES.index(name) for name in args.entropies))
    curves = {}
    lrs = {}
    times = {}
    problems = {}
    dimensions = [JointMLR(s, m).alpha.size + JointMLR(s, m).beta.size for s, m in manifolds]
    # The cost of training is taken proportional to the dimension of the manifold
    progress = Progress([((label, i), dimensions[i]) for i in manifold_indices for _ in entropy_indices
                         for _ in range(many_experiments) for label in labels])
    logging.info(f"{args.features}: manifolds {', '.join(f'M{i + 1}' for i in manifold_indices)} of dimension "
                 f"{[dimensions[i] for i in manifold_indices]}, "
                 f"{', '.join(ENTROPIES[j] for j in entropy_indices)} entropy, {many_experiments} experiments and "
                 f"{len(labels)} algorithms, {len(progress.tasks)} trainings on {n} observations")
    for i in manifold_indices:
        for j in entropy_indices:
            sigma_sq = entropies[j]
            d = {algo_class.CLASS_NAME: [] for algo_class in algorithm_classes}
            for exp_num in range(many_experiments):
                # The features of each experiment are new, since some families keep a warm start between their calls,
                # so that an experiment gives the same results whatever ran before it
                s, m = build_manifolds(args.features)[i]
                problem = JointMLR(s, m)
                np.random.seed(exp_num)
                problem.set_random_eta(sigma_sq)
                problem.set_pyx(args.support_size)
                problems[(i, j, exp_num)] = problem
                sample = JointMLRSampleIterator(problem, n, epochs, batch, random_seed=exp_num)
                # Drawn once and shared by the learning rate candidates of all the algorithms
                sample_lr = list(JointMLRSampleIterator(problem, n_lr, 1, batch, random_seed=exp_num+1))
                for algo_class in algorithm_classes:
                    model = JointMLR(s, m)
                    algo = algo_class(model)
                    lr = model.find_best_lr(problem, sample_lr, algo, verbose=not args.quiet)
                    model.fit(sample, algo, verbose=not args.quiet, lr=lr)
                    model.compute_history_metrics(problem)
                    d[algo.CLASS_NAME].append(model.history['error'])
                    lrs[(i, j, exp_num, algo.CLASS_NAME)] = lr
                    times[(i, j, exp_num, algo.CLASS_NAME)] = model.history['time']
                    logging.info(f"M{i + 1}, {ENTROPIES[j]} entropy, experiment {exp_num + 1}/{many_experiments}, "
                                 f"{algo.CLASS_NAME}: final error {model.history['error'][-1]:.4e}, learning rate "
                                 f"{'/'.join(f'{v:g}' for v in np.atleast_1d(lr))}, training "
                                 f"{model.history['time']:.1f} s ({1e6 * model.history['time'] / (n * epochs):.2f} "
                                 f"us/sample); {progress.step()}")
            curves[(i, j)] = d
            if args.output:
                save_curves(args.output, curves, lrs, times, labels, n)

    # The figure shows all the manifolds and entropies of the output, with those of previous runs
    if args.output:
        with np.load(args.output + '.npz') as z:
            cells = {(i, j): z[f'curves_{i}_{j}'] for i in range(3) for j in range(3) if f'curves_{i}_{j}' in z.files}
    else:
        cells = {cell: np.array([d[label] for label in labels]) for cell, d in curves.items()}
    missing = np.full(next(iter(cells.values())).shape, np.nan)
    # (manifold, entropy, algorithm, experiment, checkpoint)
    errors = np.array([[cells.get((i, j), missing) for j in range(3)] for i in range(3)])
    gr = np.median(errors, axis=3)
    up = np.percentile(errors, q=75, axis=3)
    dn = np.percentile(errors, q=25, axis=3)
    checkpoints = errors.shape[-1]
    x = np.arange(checkpoints) * n*epochs // checkpoints
    if args.output:
        plt.figure(figsize=(14, 9))
    plot_lines(x, gr, labels, y_labels=['M1', 'M2', 'M3'], x_labels=['High entropy', 'Medium entropy', 'Low entropy'],
               low_lines=up, high_lines=dn, file_name=args.output or '')

    logging.info(f"finished in {progress.elapsed()}")
