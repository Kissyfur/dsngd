import logging
import math
from dataclasses import dataclass
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


MIN_EXCESS_NLL = 1e-10
ITER_KEEP = 100
DEFAULT_LR_VALIDATION_SIZE = 20_000
DEFAULT_EVAL_VALIDATION_SIZE = 100_000

ENTROPY_SCENARIOS = (
    ("High entropy", 0.1),
    ("Medium entropy", 0.7),
    ("Low entropy", 1.0),
)

ALGORITHMS = (
    ("SGD", SGD_NaiveBayesEF),
    ("DSNGD", DSNGD_NaiveBayesEF),
)

PURE_FAMILY_KEYS = ("categorical", "gaussian", "poisson", "exponential")


@dataclass(frozen=True)
class EFExperimentSpec:
    key: str
    title: str
    output_name: str
    default_output_dir: str
    complexity_scenarios: tuple


def categorical(values):
    return lambda: CategoricalCoordinate(values)


def gaussian(variance=1.0):
    return lambda: GaussianKnownVarianceCoordinate(variance=variance)


def poisson():
    return PoissonCoordinate


def exponential():
    return ExponentialMeanCoordinate


def repeated_family_scenarios(factory):
    return tuple(
        (name, many_classes, tuple(factory for _ in feature_values))
        for name, many_classes, feature_values in paper_discrete_layouts()
    )


def paper_discrete_layouts():
    return (
        ("M1", 10, (10, 5)),
        ("M2", 20, (10, 5, 10, 5)),
        ("M3", 30, (10, 5, 10, 5, 10, 5)),
    )


def categorical_scenarios():
    return tuple(
        (name, many_classes, tuple(categorical(value) for value in feature_values))
        for name, many_classes, feature_values in paper_discrete_layouts()
    )


def mixed_repeated_block(repetitions):
    category_values = (10, 5, 10)
    factories = []
    for repetition in range(repetitions):
        factories.extend(
            (
                categorical(category_values[repetition % len(category_values)]),
                gaussian(),
                poisson(),
                exponential(),
            )
        )
    return tuple(factories)


def mixed_repeated_scenarios():
    return (
        ("M1", 10, mixed_repeated_block(1)),
        ("M2", 20, mixed_repeated_block(2)),
        ("M3", 30, mixed_repeated_block(3)),
    )


MIXED_COMPLEXITY_SCENARIOS = (
    ("M1", 4, (categorical(4), gaussian(), poisson())),
    ("M2", 8, (categorical(5), categorical(4), gaussian(), gaussian(), poisson(), exponential())),
    (
        "M3",
        12,
        (
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
        ),
    ),
)


FAMILY_EXPERIMENT_SPECS = {
    "categorical": EFExperimentSpec(
        key="categorical",
        title="Categorical EF",
        output_name="categorical_ef",
        default_output_dir="ef_categorical_experiment",
        complexity_scenarios=categorical_scenarios(),
    ),
    "gaussian": EFExperimentSpec(
        key="gaussian",
        title="Gaussian EF",
        output_name="gaussian_ef",
        default_output_dir="ef_gaussian_experiment",
        complexity_scenarios=repeated_family_scenarios(gaussian()),
    ),
    "poisson": EFExperimentSpec(
        key="poisson",
        title="Poisson EF",
        output_name="poisson_ef",
        default_output_dir="ef_poisson_experiment",
        complexity_scenarios=repeated_family_scenarios(poisson()),
    ),
    "exponential": EFExperimentSpec(
        key="exponential",
        title="Exponential EF",
        output_name="exponential_ef",
        default_output_dir="ef_exponential_experiment",
        complexity_scenarios=repeated_family_scenarios(exponential()),
    ),
    "mixed_repeated": EFExperimentSpec(
        key="mixed_repeated",
        title="Repeated Mixed EF",
        output_name="mixed_repeated_ef",
        default_output_dir="ef_mixed_repeated_experiment",
        complexity_scenarios=mixed_repeated_scenarios(),
    ),
    "mixed": EFExperimentSpec(
        key="mixed",
        title="Mixed EF",
        output_name="mixed_ef",
        default_output_dir="ef_mixed_experiment",
        complexity_scenarios=MIXED_COMPLEXITY_SCENARIOS,
    ),
}


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
    true_model,
    lr_train_size,
    batch,
    train_seed,
    x_lr_val,
    y_lr_val,
    progress_bar=True,
):
    optimizer = algorithm_class(model_factory())
    data = {
        "model_factory": model_factory,
        "sample_factory": lambda: NaiveBayesEFSampleIterator(
            true_model,
            epoch_length=lr_train_size,
            epochs=1,
            batch=batch,
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
                ax.set_ylabel(f"{complexity_labels[row]}\nExcess evaluation NLL")
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
    fig.savefig(output_dir / f"{output_name}_excess_evaluation_nll.png", dpi=160)
    plt.close(fig)


def save_summary(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    header = "family,complexity,entropy,experiment,algorithm,learning_rate_a,learning_rate_b,final_excess_nll"
    lines = [header]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    (output_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_curves(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    header = "family,complexity,entropy,experiment,algorithm,samples_seen,excess_evaluation_nll"
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
):
    if validation_size is not None:
        eval_validation_size = validation_size

    if output_dir is None:
        output_dir = Path("outputs") / spec.default_output_dir
    else:
        output_dir = Path(output_dir)

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
                    "Running family=%s complexity=%s entropy=%s experiment=%s",
                    spec.key,
                    complexity_name,
                    entropy_name,
                    exp_num,
                )
                true_model = build_true_model(
                    many_classes,
                    family_factories,
                    sigma,
                    seed=experiment_seed(0, row_index, col_index, exp_num),
                )
                x_lr_val, y_lr_val = collect_sample(
                    true_model,
                    lr_validation_size,
                    batch=1000,
                    seed=experiment_seed(20_000, row_index, col_index, exp_num),
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
                        true_model,
                        lr_size,
                        batch,
                        train_seed=experiment_seed(60_000, row_index, col_index, exp_num),
                        x_lr_val=x_lr_val,
                        y_lr_val=y_lr_val,
                        progress_bar=progress_bar,
                    )
                    logging.info("Selected lr for %s: a=%g, b=%g", algorithm_name, lr[0], lr[1])
                    curve = run_algorithm(
                        algorithm_class,
                        model_factory,
                        true_model,
                        lr,
                        train_size,
                        batch,
                        seed=experiment_seed(80_000, row_index, col_index, exp_num),
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
                            (spec.key, complexity_name, entropy_name, exp_num, algorithm_name, samples, value)
                        )
                    summary_rows.append(
                        (spec.key, complexity_name, entropy_name, exp_num, algorithm_name, lr[0], lr[1], curve[-1])
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
