import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.data.array_iterator import ArrayBatchIterator
from src.data.mnist import load_mnist
from src.experiments.ef_grid import (
    ALGORITHMS,
    ITER_KEEP,
    format_learning_rate,
    learning_rate_columns,
    validation_curve,
)
from src.families import CategoricalCoordinate, GaussianKnownVarianceCoordinate, PoissonCoordinate
from src.grapher import color, linestyles
from src.model.naive_bayes_ef import NaiveBayesEF


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Run EF optimizers on MNIST.")
    parser.add_argument("--data-dir", type=Path, default=Path("saved_data") / "mnist")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "ef_mnist_experiment")
    parser.add_argument(
        "--feature-family",
        choices=("gaussian", "binary-categorical", "poisson"),
        default="gaussian",
        help="Per-pixel exponential-family coordinate to use.",
    )
    parser.add_argument("--gaussian-variance", type=float, default=1.0)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--max-features", type=int, default=None, help="Keep the highest-variance pixels only.")
    parser.add_argument("--train-size", type=int, default=None, help="Training samples after LR-validation split.")
    parser.add_argument("--lr-size", type=int, default=10_000)
    parser.add_argument("--lr-validation-size", type=int, default=10_000)
    parser.add_argument("--eval-size", type=int, default=None, help="Evaluation samples from the MNIST test split.")
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(name for name, _ in ALGORITHMS),
        default=tuple(name for name, _ in ALGORITHMS),
    )
    return parser.parse_args()


def prepare_features(x_train, x_lr_val, x_eval, args):
    if args.feature_family == "gaussian":
        x_train = x_train.astype(float) / 255.0
        x_lr_val = x_lr_val.astype(float) / 255.0
        x_eval = x_eval.astype(float) / 255.0
    elif args.feature_family == "binary-categorical":
        x_train = (x_train.astype(float) / 255.0 >= args.binary_threshold).astype(int)
        x_lr_val = (x_lr_val.astype(float) / 255.0 >= args.binary_threshold).astype(int)
        x_eval = (x_eval.astype(float) / 255.0 >= args.binary_threshold).astype(int)
    elif args.feature_family == "poisson":
        x_train = x_train.astype(float)
        x_lr_val = x_lr_val.astype(float)
        x_eval = x_eval.astype(float)
    else:
        raise ValueError(f"unknown feature family {args.feature_family}")

    feature_indices = select_feature_indices(x_train, args.max_features)
    x_train = x_train[:, feature_indices]
    x_lr_val = x_lr_val[:, feature_indices]
    x_eval = x_eval[:, feature_indices]
    return x_train, x_lr_val, x_eval, make_family_factories(args, len(feature_indices)), feature_indices


def select_feature_indices(x_train, max_features):
    feature_count = x_train.shape[1]
    if max_features is None or max_features >= feature_count:
        return np.arange(feature_count)
    if max_features <= 0:
        raise ValueError("max_features must be positive")
    variances = np.var(x_train, axis=0)
    return np.sort(np.argsort(variances)[-max_features:])


def make_family_factories(args, feature_count):
    if args.feature_family == "gaussian":
        return tuple(
            lambda variance=args.gaussian_variance: GaussianKnownVarianceCoordinate(variance)
            for _ in range(feature_count)
        )
    if args.feature_family == "binary-categorical":
        return tuple(lambda: CategoricalCoordinate(2) for _ in range(feature_count))
    if args.feature_family == "poisson":
        return tuple(PoissonCoordinate for _ in range(feature_count))
    raise ValueError(f"unknown feature family {args.feature_family}")


def split_train_and_lr_validation(x, y, lr_validation_size, seed):
    if lr_validation_size <= 0 or lr_validation_size >= len(x):
        raise ValueError("lr_validation_size must be positive and smaller than the MNIST train split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    lr_indices = order[:lr_validation_size]
    train_indices = order[lr_validation_size:]
    return x[train_indices], y[train_indices], x[lr_indices], y[lr_indices]


def build_model(many_classes, family_factories):
    return NaiveBayesEF(many_classes, [factory() for factory in family_factories])


def choose_best_lr_real(
    algorithm_class,
    model_factory,
    x_train,
    y_train,
    lr_size,
    batch,
    train_seed,
    x_lr_val,
    y_lr_val,
    progress_bar=True,
):
    if lr_size > len(x_train):
        raise ValueError("lr_size must be no larger than train_size because LR search uses the training prefix")
    optimizer = algorithm_class(model_factory())
    data = {
        "model_factory": model_factory,
        "sample_factory": lambda: ArrayBatchIterator(
            x_train,
            y_train,
            batch=batch,
            epochs=1,
            random_seed=train_seed,
            max_samples=lr_size,
        ),
        "validation_curve": lambda fit_model, etas: validation_curve(fit_model, etas, x_lr_val, y_lr_val),
        "iter_keep": ITER_KEEP,
        "score_tail": 5,
    }
    return optimizer.adjust_lr_with_data(data, progress_bar=progress_bar)


def run_algorithm_real(
    algorithm_class,
    model_factory,
    x_train,
    y_train,
    train_size,
    epochs,
    batch,
    train_seed,
    lr,
    x_eval,
    y_eval,
    progress_bar=True,
):
    model = model_factory()
    optimizer = algorithm_class(model)
    train = ArrayBatchIterator(
        x_train,
        y_train,
        batch=batch,
        epochs=epochs,
        random_seed=train_seed,
        max_samples=train_size,
    )
    etas = optimizer.run(
        train,
        model.eta,
        lr=lr,
        iter_keep=ITER_KEEP,
        verbose=progress_bar,
        desc=f"{optimizer.key} MNIST training",
    )
    nll = validation_curve(model, etas, x_eval, y_eval)
    accuracy = accuracy_curve(model, etas, x_eval, y_eval)
    return nll, accuracy


def accuracy_curve(model, etas, x, y):
    values = []
    for eta in etas:
        log_probabilities = model.log_conditional_probabilities(x, eta)
        values.append(float(np.mean(np.argmax(log_probabilities, axis=1) == y)))
    return np.array(values)


def select_algorithms(names):
    requested = set(names)
    return [(name, algorithm_class) for name, algorithm_class in ALGORITHMS if name in requested]


def plot_curves(output_dir, x_axis, results, metric, ylabel, filename, log_y=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_x = x_axis.copy()
    plot_x[plot_x <= 0] = 1.0
    plt.figure(figsize=(7, 4.2))
    for name, result in results.items():
        plt.plot(
            plot_x,
            result[metric],
            label=f"{name} {format_learning_rate(result['lr'])}",
            color=color[name],
            linestyle=linestyles[name],
        )
    plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlabel("Samples seen")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()


def save_outputs(output_dir, x_axis, results):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = ["algorithm,learning_rate_0,learning_rate_1,initial_nll,final_nll,initial_accuracy,final_accuracy"]
    curve_lines = ["algorithm,samples_seen,evaluation_nll,evaluation_accuracy"]
    for name, result in results.items():
        lr_0, lr_1 = learning_rate_columns(result["lr"])
        summary_lines.append(
            f"{name},{lr_0},{lr_1},{result['nll'][0]},{result['nll'][-1]},"
            f"{result['accuracy'][0]},{result['accuracy'][-1]}"
        )
        for samples, nll, accuracy in zip(x_axis, result["nll"], result["accuracy"]):
            curve_lines.append(f"{name},{samples},{nll},{accuracy}")

    (output_dir / "summary.csv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (output_dir / "curves.csv").write_text("\n".join(curve_lines) + "\n", encoding="utf-8")
    plot_curves(output_dir, x_axis, results, "nll", "Evaluation NLL", "evaluation_nll.png", log_y=True)
    plot_curves(output_dir, x_axis, results, "accuracy", "Evaluation accuracy", "evaluation_accuracy.png")


def main():
    args = parse_args()
    x_train_full, y_train_full, x_eval, y_eval = load_mnist(args.data_dir, download=not args.no_download)
    x_train, y_train, x_lr_val, y_lr_val = split_train_and_lr_validation(
        x_train_full,
        y_train_full,
        args.lr_validation_size,
        seed=args.seed,
    )
    if args.eval_size is not None:
        x_eval = x_eval[: args.eval_size]
        y_eval = y_eval[: args.eval_size]

    x_train, x_lr_val, x_eval, family_factories, feature_indices = prepare_features(x_train, x_lr_val, x_eval, args)
    train_size = len(x_train) if args.train_size is None else min(args.train_size, len(x_train))
    if args.lr_size > train_size:
        raise ValueError("lr_size must be no larger than train_size because LR search uses the training prefix")

    many_classes = int(np.max(y_train_full)) + 1
    model_factory = lambda: build_model(many_classes, family_factories)
    algorithms = select_algorithms(args.algorithms)
    x_axis = real_samples_seen(train_size, args.batch, args.epochs, ITER_KEEP)
    results = {}

    logging.info(
        "MNIST EF experiment: family=%s train=%s lr_train=%s lr_val=%s eval=%s features=%s algorithms=%s",
        args.feature_family,
        train_size,
        args.lr_size,
        len(x_lr_val),
        len(x_eval),
        len(feature_indices),
        ",".join(name for name, _ in algorithms),
    )

    for algorithm_name, algorithm_class in algorithms:
        logging.info("Algorithm: %s", algorithm_name)
        lr = choose_best_lr_real(
            algorithm_class,
            model_factory,
            x_train,
            y_train,
            args.lr_size,
            args.batch,
            train_seed=args.seed + 1,
            x_lr_val=x_lr_val,
            y_lr_val=y_lr_val,
            progress_bar=not args.no_progress,
        )
        logging.info("Selected lr for %s: %s", algorithm_name, format_learning_rate(lr))
        nll, accuracy = run_algorithm_real(
            algorithm_class,
            model_factory,
            x_train,
            y_train,
            train_size,
            args.epochs,
            args.batch,
            train_seed=args.seed + 1,
            lr=lr,
            x_eval=x_eval,
            y_eval=y_eval,
            progress_bar=not args.no_progress,
        )
        results[algorithm_name] = {"lr": lr, "nll": nll, "accuracy": accuracy}
        save_outputs(args.output_dir, x_axis, results)
        logging.info(
            "Finished %s with final NLL %.6f and accuracy %.4f",
            algorithm_name,
            nll[-1],
            accuracy[-1],
        )

    save_outputs(args.output_dir, x_axis, results)
    print(f"Saved MNIST EF experiment outputs to {args.output_dir}")


def real_samples_seen(train_size, batch, epochs, iter_keep=ITER_KEEP):
    if train_size <= 0 or batch <= 0 or epochs <= 0:
        raise ValueError("train_size, batch, and epochs must be positive")
    effective_batch = batch if batch < train_size else train_size
    total_batches = epochs * int(np.ceil(train_size / effective_batch))
    stride = max(total_batches // iter_keep, 1)
    kept = np.arange(0, total_batches, stride) * effective_batch
    return np.concatenate([np.minimum(kept, train_size * epochs), np.array([train_size * epochs])])


if __name__ == "__main__":
    main()
