import argparse
import logging
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from src.model.joint_mlr import JointMLR
from src.model.statistic import BetaStatistic, GammaStatistic, GaussianMeanVarianceStatistic, VonMisesStatistic
from src.data.array_iterator import ArraySampleIterator
from src.data.datasets import load_covertype, load_mnist, standardize
from src.algorithms.sgd import SGD_JointMLR
from src.algorithms.dsngd import DSNGD_JointMLR
from src.algorithms.sngd import SNGD_JointMLR
from src.algorithms.adaGrad import AdaGrad_JointMLR
from src.grapher import color, linestyles
from src.progress import Progress


logging.basicConfig(level=logging.INFO)

ALGORITHMS = {algo_class.CLASS_NAME: algo_class
              for algo_class in [SGD_JointMLR, AdaGrad_JointMLR, DSNGD_JointMLR, SNGD_JointMLR]}
DEFAULTS = {'covertype': {'epochs': 3, 'n_lr': 500 * 250, 'features': 'natural'},
            'mnist': {'epochs': 20, 'n_lr': 100 * 250, 'features': 'levels'}}
FEATURES = {'covertype': ['natural', 'gaussian'], 'mnist': ['levels', 'bernoulli', 'gaussian']}
# Known results of other models, drawn on the accuracy plot. Trained without distortions, the test error of LeNet-5
# (about 60,000 parameters) stabilizes at 0.95% after about 10 of its 20 passes through the training set
# (LeCun et al., Gradient-based learning applied to document recognition, Proc. IEEE 86(11), 1998).
REFERENCES = {'mnist': [{'label': 'LeNet-5 (LeCun et al., 1998)', 'accuracy': 1. - 0.0095, 'epochs': 10}]}


def load(dataset, features, levels):
    if dataset == 'covertype' and features == 'gaussian':
        # Standardized continuous features as Gaussians, wilderness area and soil type as categorical features
        x_train, y_train, x_test, y_test = load_covertype()
        statistics = [GaussianMeanVarianceStatistic() for _ in range(10)] + [4, 40]
    elif dataset == 'covertype':
        # A family for each continuous feature according to its nature
        x_train, y_train, x_test, y_test = load_covertype(standardized=False)
        gaussian, angle, positive, index = [0, 4], [1], [2, 3, 5, 9], [6, 7, 8]
        standardize(x_train, x_test, gaussian)
        x_train[:, angle], x_test[:, angle] = np.radians(x_train[:, angle]), np.radians(x_test[:, angle])
        # Slope and distances, which can be 0, plus one and divided by their mean (the family is scale invariant)
        scale = np.mean(x_train[:, positive] + 1., axis=0)
        x_train[:, positive], x_test[:, positive] = (x_train[:, positive] + 1.) / scale, (x_test[:, positive] + 1.) / scale
        # Hillshade indices in {0, ..., 255}, mapped to (0, 1)
        x_train[:, index], x_test[:, index] = (x_train[:, index] + 0.5) / 256., (x_test[:, index] + 0.5) / 256.
        families = {**{j: GaussianMeanVarianceStatistic for j in gaussian}, **{j: VonMisesStatistic for j in angle},
                    **{j: GammaStatistic for j in positive}, **{j: BetaStatistic for j in index}}
        statistics = [families[j]() for j in range(10)] + [4, 40]
    elif features == 'levels':
        # Categorical intensity levels of each pixel
        x_train, y_train, x_test, y_test = load_mnist(encoding='levels', levels=levels)
        statistics = [levels] * x_train.shape[1]
    elif features == 'bernoulli':
        x_train, y_train, x_test, y_test = load_mnist(encoding='binary')
        statistics = [2] * x_train.shape[1]
    else:
        x_train, y_train, x_test, y_test = load_mnist()
        statistics = [GaussianMeanVarianceStatistic() for _ in range(x_train.shape[1])]
    return statistics, x_train, y_train, x_test, y_test


def evaluate(model, tx, y, etas, chunk=10):
    # tx is the statistic of the observations
    log_loss = np.concatenate([model.log_loss(None, y, etas[i:i + chunk], tx) for i in range(0, len(etas), chunk)])
    accuracy = np.concatenate([model.accuracy(None, y, etas[i:i + chunk], tx) for i in range(0, len(etas), chunk)])
    return log_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Compare the algorithms on a real data set.')
    parser.add_argument('--dataset', choices=list(FEATURES), required=True)
    parser.add_argument('--features', help='Model of the features: natural or gaussian for covertype, and levels, '
                                           'bernoulli or gaussian for mnist')
    parser.add_argument('--levels', type=int, default=13, help='Intensity levels of the pixels with --features levels')
    parser.add_argument('--epochs', type=int, help='Epochs over the training set')
    parser.add_argument('--n-lr', type=int, help='Training observations used in the learning rate search')
    parser.add_argument('--min-lr-exponent', type=int, default=-7,
                        help='Smallest initial learning rate of the search, as a power of 10 (-4 on synthetic data)')
    parser.add_argument('--max-train', type=int, help='Use only this many training observations')
    parser.add_argument('--seeds', type=int, default=5, help='Repetitions with different orders of the data')
    parser.add_argument('--algorithms', nargs='+', choices=list(ALGORITHMS), default=['SGD', 'AdaGrad', 'DSNGD'])
    parser.add_argument('--output', required=True, help='Path without extension where the results are saved')
    parser.add_argument('--quiet', action='store_true', help='Hide the training progress bars')
    args = parser.parse_args()
    args.features = args.features or DEFAULTS[args.dataset]['features']
    if args.features not in FEATURES[args.dataset]:
        parser.error(f'--features must be one of {FEATURES[args.dataset]} for {args.dataset}')
    return args


def save(file_name, epochs, results, title, references):
    np.savez(file_name + '.npz', epochs=epochs, **{f'{name}_{metric}': np.array(result[metric])
                                                 for name, result in results.items()
                                                 for metric in ('log_loss', 'accuracy', 'time')})
    lines = ['algorithm,seeds,learning_rates,test_log_loss_median,test_log_loss_q25,test_log_loss_q75,'
             'test_accuracy_median,test_accuracy_q25,test_accuracy_q75,training_seconds_median,us_per_sample_median']
    for name, result in results.items():
        log_loss, accuracy = np.array(result['log_loss'])[:, -1], np.array(result['accuracy'])[:, -1]
        lrs = ' '.join('/'.join(f'{v:g}' for v in np.atleast_1d(lr)) for lr in result['lr'])
        seconds = np.median(result['time'])
        lines.append(f'{name},{len(log_loss)},{lrs},' + ','.join(f'{v:.6f}' for v in (
            np.median(log_loss), *np.percentile(log_loss, [25, 75]),
            np.median(accuracy), *np.percentile(accuracy, [25, 75]), seconds, 1e6 * seconds / result['samples'])))
    with open(file_name + '.csv', 'w') as file:
        file.write('\n'.join(lines) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, label in zip(axes, ('log_loss', 'accuracy'), ('Test log-loss', 'Test accuracy')):
        for name, result in results.items():
            values = np.array(result[metric])
            ax.plot(epochs[1:], np.median(values, axis=0)[1:], label=name, color=color[name],
                    linestyle=linestyles[name])
            ax.fill_between(epochs[1:], np.percentile(values, 25, axis=0)[1:], np.percentile(values, 75, axis=0)[1:],
                            facecolor=color[name], alpha=0.35)
        ax.set_xscale('log')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(label)
    for reference in references:
        # Accuracy of the reference model and epochs it needs
        axes[1].axhline(reference['accuracy'], color='k', linestyle=':', linewidth=1, label=reference['label'])
        axes[1].axvline(reference['epochs'], color='k', linestyle=':', linewidth=1)
        axes[1].plot(reference['epochs'], reference['accuracy'], 'ko', markersize=4)
    axes[0].set_yscale('log')
    axes[0].legend()
    if references:
        axes[1].legend(loc='lower right')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(file_name + '.png')
    plt.close(fig)


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs or DEFAULTS[args.dataset]['epochs']
    n_lr = args.n_lr or DEFAULTS[args.dataset]['n_lr']
    batch = 250
    features, x_train, y_train, x_test, y_test = load(args.dataset, args.features, args.levels)
    s = int(np.max(y_train)) + 1

    # 10% of the training set is kept apart to choose the learning rates
    order = np.random.default_rng(0).permutation(len(y_train))
    n_validation = len(y_train) // 10
    x_val, y_val = x_train[order[:n_validation]], y_train[order[:n_validation]]
    x_fit, y_fit = x_train[order[n_validation:]], y_train[order[n_validation:]]
    if args.max_train:
        x_fit, y_fit = x_fit[:args.max_train], y_fit[:args.max_train]
    dimension = JointMLR(s, features).alpha.size + JointMLR(s, features).beta.size
    # Statistics of the validation and test observations, which are the same for all the models
    tx_val, tx_test = JointMLR(s, features).T.transform(x_val), JointMLR(s, features).T.transform(x_test)
    title = f'{args.dataset} ({args.features}): {len(y_fit)} train, {len(y_val)} validation, {len(y_test)} test, ' \
            f'{len(features)} features, {s} classes, {dimension} parameters'
    progress = Progress([((name,), 1) for _ in range(args.seeds) for name in args.algorithms])
    logging.info(f'{title}; {args.seeds} orders of the data and {len(args.algorithms)} algorithms, '
                 f'{len(progress.tasks)} trainings of {epochs} epochs')

    results = {name: {'log_loss': [], 'accuracy': [], 'lr': [], 'time': []} for name in args.algorithms}
    for seed in range(args.seeds):
        sample = ArraySampleIterator(x_fit, y_fit, epochs=epochs, batch=batch, random_seed=seed)
        sample_lr = ArraySampleIterator(x_fit, y_fit, epochs=1, batch=batch, random_seed=seed, epoch_length=n_lr)
        for name in args.algorithms:
            model = JointMLR(s, features)
            algo = ALGORITHMS[name](model)
            lr = model.find_best_lr(None, sample_lr, algo,
                                    metric=lambda etas: model.log_loss(None, y_val, etas, tx_val),
                                    min_exponent=args.min_lr_exponent, verbose=not args.quiet)
            model.fit(sample, algo, lr=lr, verbose=not args.quiet)
            log_loss, accuracy = evaluate(model, tx_test, y_test, model.history['etas'])
            results[name]['log_loss'].append(log_loss)
            results[name]['accuracy'].append(accuracy)
            results[name]['lr'].append(lr)
            results[name]['time'].append(model.history['time'])
            results[name]['samples'] = epochs * sample.epoch_length
            logging.info(f'{name}, order {seed + 1}/{args.seeds}: test log-loss {log_loss[-1]:.4f}, test accuracy '
                         f'{accuracy[-1]:.4f}, learning rate {"/".join(f"{v:g}" for v in np.atleast_1d(lr))}, '
                         f'training {model.history["time"]:.1f} s '
                         f'({1e6 * model.history["time"] / (epochs * sample.epoch_length):.2f} us/sample); '
                         f'{progress.step()}')
        # Epochs at the kept iterations of the optimizers and at the end
        kept = np.arange(0, len(sample), max(len(sample) // 100, 1))
        x = np.append(kept * batch, epochs * sample.epoch_length) / sample.epoch_length
        save(args.output, x, results, title, REFERENCES.get(args.dataset, []))

    logging.info(f'finished in {progress.elapsed()}')
