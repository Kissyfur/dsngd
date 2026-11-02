"""Summary of the experiment outputs for the paper.

Prints the final errors of the synthetic experiments and LaTeX rows for the tables of real data and computational time,
copies the figures to the folder of the manuscript and draws there the figures that group those of its appendices.
"""
import argparse
import csv
import glob
import os
import shutil

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter

from src.grapher import REFERENCES, color, linestyles

ENTROPIES = ['high', 'medium', 'low']
# The experiments of the appendices of the manuscript, grouped in a figure each, with their titles there
FAMILIES = {'discrete': 'Discrete', 'gaussian': 'Gaussian,\nknown var.', 'gaussian-mv': 'Gaussian,\nunknown var.',
            'multivariate-gaussian': 'Multivariate\nGaussian', 'poisson': 'Poisson', 'exponential': 'Exponential',
            'gamma': 'Gamma', 'beta': 'Beta', 'dirichlet': 'Dirichlet'}
MODELS = {'covertype_gaussian': 'Covertype, Gaussian', 'mnist_bernoulli': 'MNIST, Bernoulli',
          'mnist_gaussian': 'MNIST, Gaussian'}
# These figures have the width of the text of the manuscript, in inches, so that their fonts are printed at these sizes
TEXT_WIDTH = 6.27
SMALL_FIGURE = {'font.size': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': .4, 'xtick.major.width': .4,
                'ytick.major.width': .4, 'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.major.pad': 1.5,
                'ytick.major.pad': 1.5}


def synthetic(directory):
    times = {}
    for path in sorted(glob.glob(os.path.join(directory, '*.npz'))):
        name = os.path.basename(path)[:-4]
        z = np.load(path)
        labels = [str(label) for label in z['labels']]
        print(f'% {name}: final error (median over experiments)')
        for key in sorted(k for k in z.files if k.startswith('curves_')):
            i, j = map(int, key.split('_')[1:])
            final = np.median(z[key][:, :, -1], axis=1)
            print(f'%   M{i + 1}, {ENTROPIES[j]} entropy: ' + ', '.join(f'{l} {v:.3e}' for l, v in zip(labels, final)))
        if 'times' in z.files:
            # microseconds per observation in M3, median over entropies and experiments
            m3 = z['times'][z['times'][:, 0] == 2]
            times[name] = {label: np.median(m3[m3[:, 3] == a, 5]) for a, label in enumerate(labels) if len(m3)}
    return times


def real(directory):
    times = {}
    for path in sorted(glob.glob(os.path.join(directory, '*.csv'))):
        name = os.path.basename(path)[:-4]
        print(f'% {name}: test log-loss and accuracy, median [quartiles]')
        with open(path) as file:
            for row in csv.DictReader(file):
                values = {key: float(row[key]) for key in row if key.startswith(('test_', 'training_', 'us_'))}
                print(f"  & {row['algorithm']} & ${values['test_log_loss_median']:.4f}$ "
                      f"$[{values['test_log_loss_q25']:.4f}, {values['test_log_loss_q75']:.4f}]$ "
                      f"& ${values['test_accuracy_median']:.4f}$ "
                      f"$[{values['test_accuracy_q25']:.4f}, {values['test_accuracy_q75']:.4f}]$ \\\\")
                if 'us_per_sample_median' in values:
                    times.setdefault(name, {})[row['algorithm']] = values['us_per_sample_median']
    return times


def copy_figures(outputs, figures):
    os.makedirs(figures, exist_ok=True)
    for path in glob.glob(os.path.join(outputs, 'synthetic', '*.png')):
        shutil.copy(path, os.path.join(figures, 'synthetic_' + os.path.basename(path)))
    for path in glob.glob(os.path.join(outputs, 'real', '*.png')):
        shutil.copy(path, os.path.join(figures, os.path.basename(path)))


def plot_runs(ax, x, runs, name):
    # Median and quartiles of the runs, one per row, without the start of the training
    ax.fill_between(x[1:], *np.percentile(runs, [25, 75], axis=0)[:, 1:], facecolor=color[name], alpha=0.35, linewidth=0)
    ax.plot(x[1:], np.median(runs, axis=0)[1:], label=name, color=color[name], linestyle=linestyles[name], linewidth=.8)


def family_figure(directory, file_name):
    # All the problems of the families: a column per family and a row per manifold and entropy, with a scale per row
    families = [name for name in FAMILIES if os.path.exists(os.path.join(directory, name + '.npz'))]
    if not families:
        return
    with plt.rc_context(SMALL_FIGURE):
        # Two empty rows separate the entropies
        fig, axes = plt.subplots(11, len(families), figsize=(TEXT_WIDTH, 7.7), sharex=True, sharey='row', squeeze=False,
                                 gridspec_kw={'height_ratios': [1, 1, 1, .2, 1, 1, 1, .2, 1, 1, 1]})
        fig.subplots_adjust(left=.085, right=.998, top=.958, bottom=.082, wspace=.08, hspace=.1)
        for ax in np.concatenate([axes[3], axes[7]]):
            ax.set_visible(False)
        for col, name in enumerate(families):
            axes[0, col].set_title(FAMILIES[name], fontsize=6.5, pad=2.5, linespacing=.95)
            with np.load(os.path.join(directory, name + '.npz')) as z:
                labels = [str(label) for label in z['labels']]
                for key in (k for k in z.files if k.startswith('curves_')):
                    i, j = map(int, key.split('_')[1:])
                    curves = z[key]
                    x = np.arange(curves.shape[-1]) * int(z['n']) // curves.shape[-1]
                    for runs, label in zip(curves, labels):
                        plot_runs(axes[4 * j + i, col], x, runs, label)
                    axes[4 * j + i, col].set_xscale('log')
                    axes[4 * j + i, col].set_yscale('log')
        for j, entropy in enumerate(ENTROPIES):
            for i in range(3):
                axes[4 * j + i, 0].set_ylabel(f'$M_{i + 1}$', labelpad=1.5)
            top, bottom = axes[4 * j, 0].get_position(), axes[4 * j + 2, 0].get_position()
            fig.text(.008, (top.y1 + bottom.y0) / 2, f'{entropy.capitalize()} entropy', rotation=90, va='center')
        for ax in axes.flat:
            ax.minorticks_off()
        for ax in axes[-1]:
            ax.set_xticks([1e5, 1e6, 1e7], ['$10^5$', '$10^6$', '$10^7$'])
            # Inside the panel, as they would otherwise overlap those of the next one
            ax.get_xticklabels()[0].set_ha('left')
            ax.get_xticklabels()[-1].set_ha('right')
        fig.supxlabel('Observations', fontsize=7, y=.032)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=len(labels), frameon=False,
                   borderaxespad=0)
        fig.savefig(file_name, dpi=400)
        plt.close(fig)


def real_figure(directory, file_name):
    # A column per model, with its test log-loss above its test accuracy
    models = [name for name in MODELS if os.path.exists(os.path.join(directory, name + '.npz'))]
    if not models:
        return
    with plt.rc_context(SMALL_FIGURE):
        fig, axes = plt.subplots(2, len(models), figsize=(TEXT_WIDTH, 3.5), sharex='col', squeeze=False)
        fig.subplots_adjust(left=.07, right=.995, top=.94, bottom=.17, wspace=.2, hspace=.1)
        for col, name in enumerate(models):
            top, bottom = axes[:, col]
            with np.load(os.path.join(directory, name + '.npz')) as z:
                for ax, metric in zip((top, bottom), ('log_loss', 'accuracy')):
                    for key in (k for k in z.files if k.endswith('_' + metric)):
                        plot_runs(ax, z['epochs'], z[key], key[:-len(metric) - 1])
                    ax.set_xscale('log')
            for reference in REFERENCES.get(name.split('_')[0], []):
                bottom.axhline(reference['accuracy'], color='k', linestyle=':', linewidth=.8, label=reference['label'])
                bottom.axvline(reference['epochs'], color='k', linestyle=':', linewidth=.8)
                bottom.plot(reference['epochs'], reference['accuracy'], 'ko', markersize=2.5)
            top.set_yscale('log')
            # Plain numbers, as powers of ten would take the room of the panels
            top.yaxis.set_major_locator(FixedLocator([.3, .4, .6, 1, 2, 3]))
            top.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            top.set_title(MODELS[name], fontsize=7, pad=3)
            bottom.set_xlabel('Epochs', labelpad=1.5)
        axes[0, 0].set_ylabel('Test log-loss', labelpad=1.5)
        axes[1, 0].set_ylabel('Test accuracy', labelpad=1.5)
        for ax in axes.flat:
            ax.minorticks_off()
        fig.legend(*axes[1, -1].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, borderaxespad=0)
        fig.savefig(file_name, dpi=400)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize the experiment outputs for the paper.')
    parser.add_argument('--outputs', default='outputs')
    parser.add_argument('--figures', default=os.path.join('manuscript', 'figures'))
    args = parser.parse_args()
    times = synthetic(os.path.join(args.outputs, 'synthetic'))
    times.update(real(os.path.join(args.outputs, 'real')))
    print('% training time per observation (microseconds), M3 for the synthetic experiments')
    for name, values in times.items():
        print(f'%   {name}: ' + ', '.join(f'{label} {value:.2f}' for label, value in values.items()))
    copy_figures(args.outputs, args.figures)
    family_figure(os.path.join(args.outputs, 'synthetic'), os.path.join(args.figures, 'synthetic_families.png'))
    real_figure(os.path.join(args.outputs, 'real'), os.path.join(args.figures, 'real_models.png'))
    print(f'% figures copied to {args.figures}')
