import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from src.algorithms.dsngd_ef import DSNGD_NaiveBayesEF
from src.algorithms.sgd_ef import SGD_NaiveBayesEF
from src.data.ef_sample_creator import NaiveBayesEFSampleIterator
from src.families import (
    CategoricalCoordinate,
    ExponentialMeanCoordinate,
    GaussianKnownVarianceCoordinate,
    PoissonCoordinate,
)
from src.grapher import color, linestyles
from src.model.naive_bayes_ef import NaiveBayesEF


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUTPUT_DIR = Path("outputs") / "ef_mixed_experiment"
MIN_EXCESS_NLL = 1e-10
ITER_KEEP = 100


def categorical(values):
    return lambda: CategoricalCoordinate(values)


def gaussian(variance=1.0):
    return lambda: GaussianKnownVarianceCoordinate(variance=variance)


def poisson():
    return PoissonCoordinate


def exponential():
    return ExponentialMeanCoordinate


COMPLEXITY_SCENARIOS = [
    ("M1", 4, [categorical(4), gaussian(), poisson()]),
    ("M2", 8, [categorical(5), categorical(4), gaussian(), gaussian(), poisson(), exponential()]),
    (
        "M3",
        12,
        [
            categorical(6),
            categorical(5),
            categorical(4),
            gaussian(),
            gaussian(),
            gaussian(),
            poisson(),
            poisson(),
            exponential(),
            exponential(),
        ],
    ),
]

ENTROPY_SCENARIOS = [
    ("High entropy", 0.1),
    ("Medium entropy", 0.7),
    ("Low entropy", 1.0),
]

ALGORITHMS = [
    ("SGD", SGD_NaiveBayesEF),
    ("DSNGD", DSNGD_NaiveBayesEF),
]


def build_model(many_classes, family_factories):
    return NaiveBayesEF(many_classes, [factory() for factory in family_factories])


def alpha_for_priors(model, priors, beta_blocks):
    priors = np.asarray(priors, dtype=float)
    log_weights = np.log(priors)
    for class_index in range(model.many_classes):
        for family, block in zip(model.families, beta_blocks):
            log_weights[class_index] -= family.log_partition(block[:, class_index])
    return log_weights[:-1] - log_weights[-1]


def random_categorical_expectation(family, rng, sigma):
    logits = rng.normal(0.0, sigma, size=family.many_values)
    probabilities = np.exp(logits - logsumexp(logits))
    return probabilities[:-1]


def random_expectation(family, rng, sigma):
    if isinstance(family, CategoricalCoordinate):
        return random_categorical_expectation(family, rng, sigma)
    if isinstance(family, GaussianKnownVarianceCoordinate):
        return rng.normal(0.0, sigma, size=family.dim)
    if isinstance(family, PoissonCoordinate):
        return np.exp(rng.normal(0.0, sigma, size=family.dim))
    if isinstance(family, ExponentialMeanCoordinate):
        return np.exp(rng.normal(0.0, sigma, size=family.dim))
    raise TypeError(f"unsupported family {type(family).__name__}")


def build_true_model(many_classes, family_factories, sigma, seed):
    rng = np.random.default_rng(seed)
    model = build_model(many_classes, family_factories)
    beta_blocks = []
    for family in model.families:
        block = np.column_stack(
            [
                family.natural_from_expectation(random_expectation(family, rng, sigma))
                for _ in range(many_classes)
            ]
        )
        beta_blocks.append(block)

    prior_logits = rng.normal(0.0, sigma, size=many_classes)
    priors = np.exp(prior_logits - logsumexp(prior_logits))
    alpha = alpha_for_priors(model, priors, beta_blocks)
    model.set_eta((alpha, beta_blocks))
    return model


def collect_sample(model, size, batch, seed):
    sample = NaiveBayesEFSampleIterator(model, epoch_length=size, epochs=1, batch=batch, random_seed=seed)
    batches = list(sample)
    x = np.vstack([batch_x for batch_x, _ in batches])
    y = np.concatenate([batch_y for _, batch_y in batches])
    return x, y


def validation_nll(model, eta, x, y):
    log_probabilities = model.log_conditional_probabilities(x, eta)
    return -float(np.mean(log_probabilities[np.arange(len(y)), y]))


def validation_curve(model, etas, x, y):
    values = []
    for eta in etas:
        try:
            values.append(validation_nll(model, eta, x, y))
        except (OverflowError, FloatingPointError, ValueError):
            values.append(np.inf)
    return np.array(values)


def clipped_excess_curve(model, etas, x, y, true_nll):
    return np.maximum(validation_curve(model, etas, x, y) - true_nll, MIN_EXCESS_NLL)


def samples_seen(train_size, batch, iter_keep):
    if train_size <= 0 or batch <= 0:
        raise ValueError("train_size and batch must be positive")
    effective_batch = batch if batch < train_size else train_size
    sample_length = math.ceil(train_size / effective_batch)
    stride = max(sample_length // iter_keep, 1)
    kept = np.arange(0, sample_length, stride) * effective_batch
    return np.concatenate([kept, np.array([train_size])])


def choose_best_lr(algorithm_class, model_factory, true_model, lr_size, batch, seed, x_val, y_val):
    optimizer = algorithm_class(model_factory())
    data = {
        "model_factory": model_factory,
        "sample_factory": lambda: NaiveBayesEFSampleIterator(
            true_model,
            epoch_length=lr_size,
            epochs=1,
            batch=batch,
            random_seed=seed,
        ),
        "validation_curve": lambda fit_model, etas: validation_curve(fit_model, etas, x_val, y_val),
        "iter_keep": ITER_KEEP,
        "score_tail": 5,
    }
    return optimizer.adjust_lr_with_data(data, progress_bar=True)


def run_algorithm(algorithm_class, model_factory, true_model, lr, train_size, batch, seed, x_val, y_val, true_nll):
    model = model_factory()
    optimizer = algorithm_class(model)
    train = NaiveBayesEFSampleIterator(true_model, epoch_length=train_size, epochs=1, batch=batch, random_seed=seed)
    etas = optimizer.run(train, model.eta, lr=lr, iter_keep=ITER_KEEP, verbose=True, desc=f"{optimizer.key} training")
    return clipped_excess_curve(model, etas, x_val, y_val, true_nll)


def plot_grid(x, median, lower, upper, complexity_labels, entropy_labels, algorithm_labels):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_x = x.copy()
    plot_x[plot_x <= 0] = 1.0
    rows, columns = len(complexity_labels), len(entropy_labels)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 9), sharex=True)

    for row in range(rows):
        for col in range(columns):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(entropy_labels[col])
            if col == 0:
                ax.set_ylabel(f"{complexity_labels[row]}\nExcess validation NLL")
            if row == rows - 1:
                ax.set_xlabel("Samples seen")

            for alg_index, name in enumerate(algorithm_labels):
                line = median[row, col, alg_index]
                ax.plot(plot_x, line, label=name, color=color[name], linestyle=linestyles[name])
                ax.fill_between(
                    plot_x,
                    lower[row, col, alg_index],
                    upper[row, col, alg_index],
                    facecolor=color[name],
                    alpha=0.25,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(algorithm_labels))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUTPUT_DIR / "mixed_ef_excess_validation_nll.png", dpi=160)
    plt.close(fig)


def save_summary(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = "complexity,entropy,experiment,algorithm,learning_rate_a,learning_rate_b,final_excess_nll"
    lines = [header]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    (OUTPUT_DIR / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_curves(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = "complexity,entropy,experiment,algorithm,samples_seen,excess_validation_nll"
    lines = [header]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    (OUTPUT_DIR / "curves.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    train_size = 10_000_000
    batch = 250
    lr_size = 500 * batch
    validation_size = 20_000
    many_experiments = 1

    x_axis = samples_seen(train_size, batch, ITER_KEEP)
    algorithm_labels = [name for name, _ in ALGORITHMS]
    median = np.zeros((len(COMPLEXITY_SCENARIOS), len(ENTROPY_SCENARIOS), len(ALGORITHMS), len(x_axis)))
    lower = np.zeros_like(median)
    upper = np.zeros_like(median)
    summary_rows = []
    curve_rows = []

    for row_index, (complexity_name, many_classes, family_factories) in enumerate(COMPLEXITY_SCENARIOS):
        model_factory = lambda mc=many_classes, ff=family_factories: build_model(mc, ff)
        for col_index, (entropy_name, sigma) in enumerate(ENTROPY_SCENARIOS):
            curves_by_algorithm = {name: [] for name, _ in ALGORITHMS}
            for exp_num in range(many_experiments):
                logging.info(
                    "Running scenario complexity=%s entropy=%s experiment=%s",
                    complexity_name,
                    entropy_name,
                    exp_num,
                )
                true_model = build_true_model(many_classes, family_factories, sigma, seed=10_000 * row_index + 100 * col_index + exp_num)
                x_val, y_val = collect_sample(true_model, validation_size, batch=1000, seed=20_000 + exp_num)
                true_nll = validation_nll(true_model, true_model.eta, x_val, y_val)
                logging.info("True model validation NLL: %.6f", true_nll)

                for algorithm_name, algorithm_class in ALGORITHMS:
                    logging.info("Algorithm: %s", algorithm_name)
                    lr = choose_best_lr(
                        algorithm_class,
                        model_factory,
                        true_model,
                        lr_size,
                        batch,
                        seed=30_000 + exp_num,
                        x_val=x_val,
                        y_val=y_val,
                    )
                    logging.info("Selected lr for %s: a=%g, b=%g", algorithm_name, lr[0], lr[1])
                    curve = run_algorithm(
                        algorithm_class,
                        model_factory,
                        true_model,
                        lr,
                        train_size,
                        batch,
                        seed=40_000 + exp_num,
                        x_val=x_val,
                        y_val=y_val,
                        true_nll=true_nll,
                    )
                    curves_by_algorithm[algorithm_name].append(curve)
                    if len(curve) != len(x_axis):
                        raise ValueError(f"curve length {len(curve)} does not match x-axis length {len(x_axis)}")
                    for samples, value in zip(x_axis, curve):
                        curve_rows.append(
                            (complexity_name, entropy_name, exp_num, algorithm_name, samples, value)
                        )
                    summary_rows.append(
                        (complexity_name, entropy_name, exp_num, algorithm_name, lr[0], lr[1], curve[-1])
                    )
                    logging.info("Finished %s with final excess validation NLL %.6g", algorithm_name, curve[-1])

            for alg_index, algorithm_name in enumerate(algorithm_labels):
                curves = np.array(curves_by_algorithm[algorithm_name])
                median[row_index, col_index, alg_index] = np.median(curves, axis=0)
                lower[row_index, col_index, alg_index] = np.percentile(curves, q=25, axis=0)
                upper[row_index, col_index, alg_index] = np.percentile(curves, q=75, axis=0)

    plot_grid(
        x_axis,
        median,
        lower,
        upper,
        [scenario[0] for scenario in COMPLEXITY_SCENARIOS],
        [scenario[0] for scenario in ENTROPY_SCENARIOS],
        algorithm_labels,
    )
    save_summary(summary_rows)
    save_curves(curve_rows)
    print(f"Saved mixed EF experiment outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
