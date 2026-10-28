"""Summary of the experiment outputs for the paper.

Prints the final errors of the synthetic experiments and LaTeX rows for the tables of real data and computational time,
and copies the figures to the folder of the manuscript.
"""
import argparse
import csv
import glob
import os
import shutil

import numpy as np

ENTROPIES = ['high', 'medium', 'low']


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
    print(f'% figures copied to {args.figures}')
