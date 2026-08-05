import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.adagrad_ef import AdaGrad_NaiveBayesEF
from src.algorithms.dsngd_ef import DSNGD_NaiveBayesEF
from src.algorithms.sgd_ef import SGD_NaiveBayesEF
from src.data.array_iterator import ArrayBatchIterator
from src.data.ef_sample_creator import NaiveBayesEFSampleIterator
from src.families import (
    ExponentialMeanCoordinate,
    GaussianUnknownVarianceCoordinate,
    MultivariateGaussianCoordinate,
)
from src.grapher import color, linestyles
from src.experiments.ef_specs import MATCHED_SAMPLER
from src.model.naive_bayes_ef import NaiveBayesEF


MIN_EXCESS_NLL = 1e-10
ITER_KEEP = 100
DEFAULT_EVAL_VALIDATION_SIZE = 100_000
DEFAULT_LR_VALIDATION_SIZE = DEFAULT_EVAL_VALIDATION_SIZE

ENTROPY_SCENARIOS = (
    ("High entropy", 0.1),
    ("Medium entropy", 0.7),
    ("Low entropy", 1.0),
)

ALGORITHMS = (
    ("SGD", SGD_NaiveBayesEF),
    ("AdaGrad", AdaGrad_NaiveBayesEF),
    ("DSNGD", DSNGD_NaiveBayesEF),
)


def build_model(many_classes, family_factories):
    return NaiveBayesEF(many_classes, [factory() for factory in family_factories])


def random_natural_block(family, many_classes, rng, sigma):
    neutral = family.natural_from_expectation(family.initial_expectation())
    natural_by_class = neutral + rng.normal(0.0, sigma, size=(many_classes, family.dim))
    return fold_to_natural_domain(family, natural_by_class).T


def fold_to_natural_domain(family, natural_parameter):
    theta = np.asarray(natural_parameter, dtype=float).copy()
    if isinstance(family, ExponentialMeanCoordinate):
        theta[..., 0] = -np.maximum(np.abs(theta[..., 0]), family.natural_margin)
        return theta
    if isinstance(family, GaussianUnknownVarianceCoordinate):
        theta[..., 1] = -np.maximum(np.abs(theta[..., 1]), family.natural_margin)
        return theta
    if isinstance(family, MultivariateGaussianCoordinate):
        h = theta[..., : family.event_dim]
        a = theta[..., family.event_dim :].reshape(theta.shape[:-1] + (family.event_dim, family.event_dim))
        a = 0.5 * (a + np.swapaxes(a, -1, -2))
        eigenvalues, eigenvectors = np.linalg.eigh(a)
        eigenvalues = -np.maximum(np.abs(eigenvalues), family.natural_margin)
        a = np.matmul(eigenvectors * eigenvalues[..., None, :], np.swapaxes(eigenvectors, -1, -2))
        return np.concatenate((h, a.reshape(theta.shape[:-1] + (family.event_dim * family.event_dim,))), axis=-1)
    return theta


def build_matched_true_model(many_classes, family_factories, sigma, seed):
    rng = np.random.default_rng(seed)
    model = build_model(many_classes, family_factories)
    alpha = rng.normal(0.0, sigma, size=many_classes - 1)
    beta_blocks = [
        random_natural_block(family, many_classes, rng, sigma)
        for family in model.families
    ]
    model.set_eta((alpha, beta_blocks))
    return model


def build_true_model(many_classes, family_factories, sigma, seed, sampler=MATCHED_SAMPLER):
    if sampler == MATCHED_SAMPLER:
        return build_matched_true_model(many_classes, family_factories, sigma, seed)
    raise ValueError(f"unsupported synthetic sampler: {sampler}")


def collect_sample(model, size, batch, seed):
    sample = NaiveBayesEFSampleIterator(model, epoch_length=size, epochs=1, batch=batch, random_seed=seed)
    batches = list(sample)
    x = np.vstack([batch_x for batch_x, _ in batches])
    y = np.concatenate([batch_y for _, batch_y in batches])
    return x, y


def experiment_seed(base, row_index, col_index, exp_num):
    return base + 10_000 * row_index + 100 * col_index + exp_num


def validation_nll(model, eta, x, y):
    log_probabilities = model.log_conditional_probabilities(x, eta)
    return -float(np.mean(log_probabilities[np.arange(len(y)), y]))


def validation_curve(model, etas, x, y):
    values = []
    for eta in etas:
        try:
            value = validation_nll(model, eta, x, y)
        except (OverflowError, FloatingPointError, ValueError):
            value = np.inf
        if not np.isfinite(value):
            value = np.inf
        values.append(value)
    return np.array(values)


def clipped_excess_curve(model, etas, x, y, true_nll):
    return np.maximum(validation_curve(model, etas, x, y) - true_nll, MIN_EXCESS_NLL)


def learning_rate_columns(lr):
    values = np.asarray(lr, dtype=float).reshape(-1)
    if len(values) == 1:
        return values[0], ""
    return values[0], values[1]


def format_learning_rate(lr):
    values = np.asarray(lr, dtype=float).reshape(-1)
    if len(values) == 1:
        return f"gamma={values[0]:g}"
    return f"a={values[0]:g}, b={values[1]:g}"


def samples_seen(train_size, batch, iter_keep=ITER_KEEP):
    if train_size <= 0 or batch <= 0:
        raise ValueError("train_size and batch must be positive")
    effective_batch = batch if batch < train_size else train_size
    sample_length = math.ceil(train_size / effective_batch)
    stride = max(sample_length // iter_keep, 1)
    kept = np.arange(0, sample_length, stride) * effective_batch
    return np.concatenate([kept, np.array([train_size])])


def choose_best_lr(
    algorithm_class,
    model_factory,
    x_lr_train,
    y_lr_train,
    batch,
    train_seed,
    x_lr_val,
    y_lr_val,
    progress_bar=True,
):
    optimizer = algorithm_class(model_factory())
    data = {
        "model_factory": model_factory,
        "sample_factory": lambda: ArrayBatchIterator(
            x_lr_train,
            y_lr_train,
            epochs=1,
            batch=batch,
            shuffle=False,
            random_seed=train_seed,
        ),
        "validation_curve": lambda fit_model, etas: validation_curve(fit_model, etas, x_lr_val, y_lr_val),
        "iter_keep": ITER_KEEP,
        "score_tail": 5,
    }
    return optimizer.adjust_lr_with_data(data, progress_bar=progress_bar)


def run_algorithm(
    algorithm_class,
    model_factory,
    true_model,
    lr,
    train_size,
    batch,
    seed,
    x_eval,
    y_eval,
    true_nll,
    progress_bar=True,
):
    model = model_factory()
    optimizer = algorithm_class(model)
    train = NaiveBayesEFSampleIterator(true_model, epoch_length=train_size, epochs=1, batch=batch, random_seed=seed)
    etas = optimizer.run(
        train,
        model.eta,
        lr=lr,
        iter_keep=ITER_KEEP,
        verbose=progress_bar,
        desc=f"{optimizer.key} training",
    )
    return clipped_excess_curve(model, etas, x_eval, y_eval, true_nll)


def plot_grid(output_dir, output_name, x, median, lower, upper, complexity_labels, entropy_labels, algorithm_labels):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_mask = x > 0
    plot_x = x[plot_mask]
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
                ax.set_ylabel(f"{complexity_labels[row]}\nExcess evaluation NLL")
            if row == rows - 1:
                ax.set_xlabel("Samples seen")

            for alg_index, name in enumerate(algorithm_labels):
                line = median[row, col, alg_index]
                ax.plot(plot_x, line[plot_mask], label=name, color=color[name], linestyle=linestyles[name])
                ax.fill_between(
                    plot_x,
                    lower[row, col, alg_index][plot_mask],
                    upper[row, col, alg_index][plot_mask],
                    facecolor=color[name],
                    alpha=0.25,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(algorithm_labels))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_dir / f"{output_name}_excess_evaluation_nll.png", dpi=160)
    plt.close(fig)


def save_summary(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    header = "family,sampler,complexity,entropy,experiment,algorithm,learning_rate_0,learning_rate_1,final_excess_nll"
    lines = [header]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    (output_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_curves(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    header = "family,sampler,complexity,entropy,experiment,algorithm,samples_seen,excess_evaluation_nll"
    lines = [header]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    (output_dir / "curves.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_grid_experiment(
    spec,
    output_dir=None,
    train_size=10_000_000,
    batch=250,
    lr_size=500 * 250,
    validation_size=None,
    many_experiments=1,
    progress_bar=True,
    lr_validation_size=DEFAULT_LR_VALIDATION_SIZE,
    eval_validation_size=DEFAULT_EVAL_VALIDATION_SIZE,
    sampler=None,
):
    if validation_size is not None:
        eval_validation_size = validation_size
    if lr_size > train_size:
        raise ValueError("lr_size must be no larger than train_size because LR search uses the training prefix")

    if output_dir is None:
        output_dir = Path("outputs") / spec.default_output_dir
    else:
        output_dir = Path(output_dir)
    sampler = spec.sampler if sampler is None else sampler

    x_axis = samples_seen(train_size, batch, ITER_KEEP)
    algorithm_labels = [name for name, _ in ALGORITHMS]
    median = np.zeros((len(spec.complexity_scenarios), len(ENTROPY_SCENARIOS), len(ALGORITHMS), len(x_axis)))
    lower = np.zeros_like(median)
    upper = np.zeros_like(median)
    summary_rows = []
    curve_rows = []

    for row_index, (complexity_name, many_classes, family_factories) in enumerate(spec.complexity_scenarios):
        model_factory = lambda mc=many_classes, ff=family_factories: build_model(mc, ff)
        for col_index, (entropy_name, sigma) in enumerate(ENTROPY_SCENARIOS):
            curves_by_algorithm = {name: [] for name, _ in ALGORITHMS}
            for exp_num in range(many_experiments):
                logging.info(
                    "Running family=%s sampler=%s complexity=%s entropy=%s experiment=%s",
                    spec.key,
                    sampler,
                    complexity_name,
                    entropy_name,
                    exp_num,
                )
                true_model = build_true_model(
                    many_classes,
                    family_factories,
                    sigma,
                    seed=experiment_seed(0, row_index, col_index, exp_num),
                    sampler=sampler,
                )
                train_seed = experiment_seed(60_000, row_index, col_index, exp_num)
                x_lr_val, y_lr_val = collect_sample(
                    true_model,
                    lr_validation_size,
                    batch=1000,
                    seed=experiment_seed(20_000, row_index, col_index, exp_num),
                )
                x_lr_train, y_lr_train = collect_sample(
                    true_model,
                    lr_size,
                    batch=batch,
                    seed=train_seed,
                )
                x_eval, y_eval = collect_sample(
                    true_model,
                    eval_validation_size,
                    batch=1000,
                    seed=experiment_seed(40_000, row_index, col_index, exp_num),
                )
                true_nll = validation_nll(true_model, true_model.eta, x_eval, y_eval)
                logging.info("True model evaluation NLL on %s samples: %.6f", eval_validation_size, true_nll)

                for algorithm_name, algorithm_class in ALGORITHMS:
                    logging.info("Algorithm: %s", algorithm_name)
                    lr = choose_best_lr(
                        algorithm_class,
                        model_factory,
                        x_lr_train,
                        y_lr_train,
                        batch,
                        train_seed=train_seed,
                        x_lr_val=x_lr_val,
                        y_lr_val=y_lr_val,
                        progress_bar=progress_bar,
                    )
                    logging.info("Selected lr for %s: %s", algorithm_name, format_learning_rate(lr))
                    curve = run_algorithm(
                        algorithm_class,
                        model_factory,
                        true_model,
                        lr,
                        train_size,
                        batch,
                        seed=train_seed,
                        x_eval=x_eval,
                        y_eval=y_eval,
                        true_nll=true_nll,
                        progress_bar=progress_bar,
                    )
                    curves_by_algorithm[algorithm_name].append(curve)
                    if len(curve) != len(x_axis):
                        raise ValueError(f"curve length {len(curve)} does not match x-axis length {len(x_axis)}")
                    for samples, value in zip(x_axis, curve):
                        curve_rows.append(
                            (spec.key, sampler, complexity_name, entropy_name, exp_num, algorithm_name, samples, value)
                        )
                    summary_rows.append(
                        (
                            spec.key,
                            sampler,
                            complexity_name,
                            entropy_name,
                            exp_num,
                            algorithm_name,
                            *learning_rate_columns(lr),
                            curve[-1],
                        )
                    )
                    logging.info("Finished %s with final excess evaluation NLL %.6g", algorithm_name, curve[-1])
                    save_summary(output_dir, summary_rows)
                    save_curves(output_dir, curve_rows)

            for alg_index, algorithm_name in enumerate(algorithm_labels):
                curves = np.array(curves_by_algorithm[algorithm_name])
                median[row_index, col_index, alg_index] = np.median(curves, axis=0)
                lower[row_index, col_index, alg_index] = np.percentile(curves, q=25, axis=0)
                upper[row_index, col_index, alg_index] = np.percentile(curves, q=75, axis=0)

    plot_grid(
        output_dir,
        spec.output_name,
        x_axis,
        median,
        lower,
        upper,
        [scenario[0] for scenario in spec.complexity_scenarios],
        [scenario[0] for scenario in ENTROPY_SCENARIOS],
        algorithm_labels,
    )
    save_summary(output_dir, summary_rows)
    save_curves(output_dir, curve_rows)
    print(f"Saved {spec.title} experiment outputs to {output_dir}")
