from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.data.ef_sample_creator import NaiveBayesEFSampleIterator
from src.experiments.ef_grid import (
    ALGORITHMS,
    choose_best_lr,
    collect_sample,
    format_learning_rate,
    learning_rate_columns,
    samples_seen,
    validation_curve,
    validation_nll,
)
from src.families import CategoricalCoordinate, GaussianKnownVarianceCoordinate
from src.grapher import color, linestyles
from src.model.naive_bayes_ef import NaiveBayesEF


OUTPUT_DIR = Path("outputs") / "ef_small_comparison"
LR_VALIDATION_SIZE = 3_000
EVAL_VALIDATION_SIZE = 100_000


def build_model():
    return NaiveBayesEF(
        3,
        [
            CategoricalCoordinate(3),
            GaussianKnownVarianceCoordinate(variance=1.0),
            GaussianKnownVarianceCoordinate(variance=1.0),
        ],
    )


def alpha_for_priors(model, priors, beta_blocks):
    priors = np.asarray(priors, dtype=float)
    log_weights = np.log(priors)
    for class_index in range(model.many_classes):
        for family, block in zip(model.families, beta_blocks):
            log_weights[class_index] -= family.log_partition(block[:, class_index])
    return log_weights[:-1] - log_weights[-1]


def build_true_model():
    model = build_model()
    categorical = model.families[0]
    category_probabilities = [
        np.array([0.65, 0.25]),
        np.array([0.15, 0.65]),
        np.array([0.20, 0.15]),
    ]
    beta_categorical = np.column_stack(
        [categorical.natural_from_expectation(probability) for probability in category_probabilities]
    )
    beta_gaussian_1 = np.array([[-1.4, 0.0, 1.4]])
    beta_gaussian_2 = np.array([[0.9, -1.1, 0.6]])
    beta_blocks = [beta_categorical, beta_gaussian_1, beta_gaussian_2]
    alpha = alpha_for_priors(model, priors=[0.35, 0.35, 0.30], beta_blocks=beta_blocks)
    model.set_eta((alpha, beta_blocks))
    return model


def run_candidate(algorithm_class, true_model, lr, train_size, batch, train_seed, x_val, y_val):
    model = build_model()
    optimizer = algorithm_class(model)
    train = NaiveBayesEFSampleIterator(true_model, epoch_length=train_size, epochs=1, batch=batch, random_seed=train_seed)
    etas = optimizer.run(train, model.eta, lr=lr, iter_keep=len(train))
    return validation_curve(model, etas, x_val, y_val)


def run_with_selected_lr(algorithm_class, selected_lr, true_model, train_size, batch, train_seed, x_val, y_val):
    curve = run_candidate(algorithm_class, true_model, selected_lr, train_size, batch, train_seed, x_val, y_val)
    return {"lr": selected_lr, "curve": curve}


def save_curves(samples_seen, results, true_nll):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_mask = samples_seen > 0
    plot_x = samples_seen[plot_mask]

    plt.figure(figsize=(7, 4.2))
    for name, result in results.items():
        lr_text = format_learning_rate(result["lr"])
        plt.plot(
            plot_x,
            result["curve"][plot_mask],
            label=f"{name} {lr_text}",
            color=color[name],
            linestyle=linestyles[name],
        )
    plt.axhline(true_nll, color="black", linestyle=":", label="true model")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Samples seen")
    plt.ylabel("Evaluation negative log-likelihood")
    plt.title("Small mixed EF classification problem")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "evaluation_nll.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    for name, result in results.items():
        excess = np.maximum(result["curve"] - true_nll, 1e-8)
        lr_text = format_learning_rate(result["lr"])
        plt.plot(
            plot_x,
            excess[plot_mask],
            label=f"{name} {lr_text}",
            color=color[name],
            linestyle=linestyles[name],
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Samples seen")
    plt.ylabel("Evaluation NLL gap over true model (clipped)")
    plt.title("Convergence on evaluation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "excess_evaluation_nll.png", dpi=160)
    plt.close()


def save_summary(samples_seen, results, true_nll):
    lines = ["algorithm,learning_rate_0,learning_rate_1,initial_nll,final_nll,true_model_nll"]
    for name, result in results.items():
        lr_0, lr_1 = learning_rate_columns(result["lr"])
        lines.append(
            f"{name},{lr_0},{lr_1},{result['curve'][0]},{result['curve'][-1]},{true_nll}"
        )
    (OUTPUT_DIR / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    curves = np.column_stack([samples_seen] + [result["curve"] for result in results.values()])
    header = "samples_seen," + ",".join(results.keys())
    np.savetxt(OUTPUT_DIR / "curves.csv", curves, delimiter=",", header=header, comments="")


def main():
    true_model = build_true_model()
    train_size = 3000
    lr_size = 750
    batch = 50
    if lr_size > train_size:
        raise ValueError("lr_size must be no larger than train_size because LR search uses the training prefix")
    train_seed = 7
    x_lr_val, y_lr_val = collect_sample(true_model, size=LR_VALIDATION_SIZE, batch=500, seed=123)
    x_lr_train, y_lr_train = collect_sample(true_model, size=lr_size, batch=batch, seed=train_seed)
    x_eval, y_eval = collect_sample(true_model, size=EVAL_VALIDATION_SIZE, batch=1000, seed=456)
    true_nll = validation_nll(true_model, true_model.eta, x_eval, y_eval)
    x_axis = samples_seen(train_size, batch)

    results = {}
    for algorithm_name, algorithm_class in ALGORITHMS:
        selected_lr = choose_best_lr(
            algorithm_class,
            build_model,
            x_lr_train,
            y_lr_train,
            batch,
            train_seed,
            x_lr_val,
            y_lr_val,
            progress_bar=False,
        )
        results[algorithm_name] = run_with_selected_lr(
            algorithm_class,
            selected_lr,
            true_model,
            train_size,
            batch,
            train_seed,
            x_eval,
            y_eval,
        )

    save_curves(x_axis, results, true_nll)
    save_summary(x_axis, results, true_nll)

    print(f"Saved results to {OUTPUT_DIR}")
    print(f"True model evaluation NLL: {true_nll:.6f}")
    for name, result in results.items():
        print(
            f"{name}: {format_learning_rate(result['lr'])}, "
            f"initial={result['curve'][0]:.6f}, final={result['curve'][-1]:.6f}"
        )


if __name__ == "__main__":
    main()
