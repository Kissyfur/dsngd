import argparse
import logging
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from src.model.joint_mlr import JointMLR
from src.model.statistic import GaussianMeanVarianceStatistic
from src.data.array_iterator import ArraySampleIterator
from src.data.datasets import load_covertype, load_mnist
from src.algorithms.sgd import SGD_JointMLR
from src.algorithms.dsngd import DSNGD_JointMLR
from src.algorithms.sngd import SNGD_JointMLR
from src.algorithms.adaGrad import AdaGrad_JointMLR
from src.grapher import color, linestyles


logging.basicConfig(level=logging.INFO)

ALGORITHMS = {algo_class.CLASS_NAME: algo_class
              for algo_class in [SGD_JointMLR, AdaGrad_JointMLR, DSNGD_JointMLR, SNGD_JointMLR]}
DEFAULTS = {'covertype': {'epochs': 3, 'n_lr': 500 * 250}, 'mnist': {'epochs': 10, 'n_lr': 100 * 250}}


def load(dataset, mnist_features):
    if dataset == 'covertype':
        # 10 continuous features, wilderness area and soil type
        x_train, y_train, x_test, y_test = load_covertype()
        features = [GaussianMeanVarianceStatistic() for _ in range(10)] + [4, 40]
    else:
        binary = mnist_features == 'bernoulli'
        x_train, y_train, x_test, y_test = load_mnist(binary=binary)
        features = [2 if binary else GaussianMeanVarianceStatistic() for _ in range(x_train.shape[1])]
    return features, x_train, y_train, x_test, y_test


def evaluate(model, x, y, etas, chunk=10):
    log_loss = np.concatenate([model.log_loss(x, y, etas[i:i + chunk]) for i in range(0, len(etas), chunk)])
    accuracy = np.concatenate([model.accuracy(x, y, etas[i:i + chunk]) for i in range(0, len(etas), chunk)])
    return log_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Compare the algorithms on a real data set.')
    parser.add_argument('--dataset', choices=['covertype', 'mnist'], required=True)
    parser.add_argument('--mnist-features', choices=['gaussian', 'bernoulli'], default='gaussian')
    parser.add_argument('--epochs', type=int, help='Epochs over the training set')
    parser.add_argument('--n-lr', type=int, help='Training observations used in the learning rate search')
    parser.add_argument('--max-train', type=int, help='Use only this many training observations')
    parser.add_argument('--seeds', type=int, default=5, help='Repetitions with different orders of the data')
    parser.add_argument('--algorithms', nargs='+', choices=list(ALGORITHMS), default=['SGD', 'AdaGrad', 'DSNGD'])
    parser.add_argument('--output', required=True, help='Path without extension where the results are saved')
    parser.add_argument('--quiet', action='store_true', help='Hide the training progress bars')
    return parser.parse_args()


def save(file_name, x, results, title):
    np.savez(file_name + '.npz', x=x, **{f'{name}_{metric}': np.array(result[metric])
                                       for name, result in results.items() for metric in ('log_loss', 'accuracy')})
    lines = ['algorithm,seeds,learning_rates,test_log_loss_median,test_log_loss_q25,test_log_loss_q75,'
             'test_accuracy_median,test_accuracy_q25,test_accuracy_q75']
    for name, result in results.items():
        log_loss, accuracy = np.array(result['log_loss'])[:, -1], np.array(result['accuracy'])[:, -1]
        lrs = ' '.join(str(np.round(np.atleast_1d(lr), 6).tolist()) for lr in result['lr'])
        lines.append(f'{name},{len(log_loss)},{lrs},' + ','.join(f'{v:.6f}' for v in (
            np.median(log_loss), *np.percentile(log_loss, [25, 75]),
            np.median(accuracy), *np.percentile(accuracy, [25, 75]))))
    with open(file_name + '.csv', 'w') as file:
        file.write('\n'.join(lines) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, label in zip(axes, ('log_loss', 'accuracy'), ('Test log-loss', 'Test accuracy')):
        for name, result in results.items():
            values = np.array(result[metric])
            ax.plot(x[1:], np.median(values, axis=0)[1:], label=name, color=color[name], linestyle=linestyles[name])
            ax.fill_between(x[1:], np.percentile(values, 25, axis=0)[1:], np.percentile(values, 75, axis=0)[1:],
                            facecolor=color[name], alpha=0.35)
        ax.set_xscale('log')
        ax.set_xlabel('Samples')
        ax.set_ylabel(label)
    axes[0].set_yscale('log')
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(file_name + '.png')
    plt.close(fig)


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs or DEFAULTS[args.dataset]['epochs']
    n_lr = args.n_lr or DEFAULTS[args.dataset]['n_lr']
    batch = 250
    features, x_train, y_train, x_test, y_test = load(args.dataset, args.mnist_features)
    s = int(np.max(y_train)) + 1

    # 10% of the training set is kept apart to choose the learning rates
    order = np.random.default_rng(0).permutation(len(y_train))
    n_validation = len(y_train) // 10
    x_val, y_val = x_train[order[:n_validation]], y_train[order[:n_validation]]
    x_fit, y_fit = x_train[order[n_validation:]], y_train[order[n_validation:]]
    if args.max_train:
        x_fit, y_fit = x_fit[:args.max_train], y_fit[:args.max_train]
    title = f'{args.dataset}: {len(y_fit)} train, {len(y_val)} validation, {len(y_test)} test, ' \
            f'{len(features)} features, {s} classes, {epochs} epochs'
    logging.info(title)

    results = {name: {'log_loss': [], 'accuracy': [], 'lr': []} for name in args.algorithms}
    for seed in range(args.seeds):
        sample = ArraySampleIterator(x_fit, y_fit, epochs=epochs, batch=batch, random_seed=seed)
        sample_lr = ArraySampleIterator(x_fit, y_fit, epochs=1, batch=batch, random_seed=seed, epoch_length=n_lr)
        for name in args.algorithms:
            model = JointMLR(s, features)
            algo = ALGORITHMS[name](model)
            lr = model.find_best_lr(None, sample_lr, algo, metric=lambda etas: model.log_loss(x_val, y_val, etas))
            model.fit(sample, algo, lr=lr, verbose=not args.quiet)
            log_loss, accuracy = evaluate(model, x_test, y_test, model.history['etas'])
            results[name]['log_loss'].append(log_loss)
            results[name]['accuracy'].append(accuracy)
            results[name]['lr'].append(lr)
            logging.info(f'{name} seed {seed}: test log-loss {log_loss[-1]:.4f}, test accuracy {accuracy[-1]:.4f}')
        # Samples seen at the kept iterations of the optimizers and at the end
        kept = np.arange(0, len(sample), max(len(sample) // 100, 1))
        x = np.append(kept * batch, epochs * sample.epoch_length)
        save(args.output, x, results, title)

    print("finished")
