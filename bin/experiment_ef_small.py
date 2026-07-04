from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.dsngd_ef import DSNGD_NaiveBayesEF
from src.algorithms.sgd_ef import SGD_NaiveBayesEF
from src.data.ef_sample_creator import NaiveBayesEFSampleIterator
from src.families import CategoricalCoordinate, GaussianKnownVarianceCoordinate
from src.model.naive_bayes_ef import NaiveBayesEF


OUTPUT_DIR = Path("outputs") / "ef_small_comparison"


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


def collect_sample(model, size, batch, seed):
    iterator = NaiveBayesEFSampleIterator(model, epoch_length=size, epochs=1, batch=batch, random_seed=seed)
    batches = list(iterator)
    x = np.vstack([batch_x for batch_x, _ in batches])
    y = np.concatenate([batch_y for _, batch_y in batches])
    return x, y


def validation_nll(model, eta, x, y):
    log_probabilities = model.log_conditional_probabilities(x, eta)
    return -float(np.mean(log_probabilities[np.arange(len(y)), y]))


def validation_curve(model, etas, x, y):
    return np.array([validation_nll(model, eta, x, y) for eta in etas])


def run_candidate(algorithm_class, true_model, lr, train_size, batch, train_seed, x_val, y_val):
    model = build_model()
    optimizer = algorithm_class(model)
    train = NaiveBayesEFSampleIterator(true_model, epoch_length=train_size, epochs=1, batch=batch, random_seed=train_seed)
    etas = optimizer.run(train, model.eta, lr=lr, iter_keep=len(train))
    return validation_curve(model, etas, x_val, y_val)


def choose_best_run(algorithm_class, candidates, true_model, train_size, batch, train_seed, x_val, y_val):
    best = None
    for lr in candidates:
        curve = run_candidate(algorithm_class, true_model, lr, train_size, batch, train_seed, x_val, y_val)
        result = {"lr": lr, "curve": curve}
        if best is None or curve[-1] < best["curve"][-1]:
            best = result
    return best


def save_curves(samples_seen, results, true_nll):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.2))
    for name, result in results.items():
        plt.plot(samples_seen, result["curve"], label=f"{name} lr={result['lr'][0]:g}")
    plt.axhline(true_nll, color="black", linestyle=":", label="true model")
    plt.xlabel("Samples seen")
    plt.ylabel("Validation negative log-likelihood")
    plt.title("Small mixed EF classification problem")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_nll.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    for name, result in results.items():
        excess = np.maximum(result["curve"] - true_nll, 1e-8)
        plt.semilogy(samples_seen, excess, label=f"{name} lr={result['lr'][0]:g}")
    plt.xlabel("Samples seen")
    plt.ylabel("Excess validation NLL over true model")
    plt.title("Convergence on validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "excess_validation_nll.png", dpi=160)
    plt.close()


def save_summary(samples_seen, results, true_nll):
    lines = ["algorithm,learning_rate,initial_nll,final_nll,true_model_nll"]
    for name, result in results.items():
        lines.append(
            f"{name},{result['lr'][0]},{result['curve'][0]},{result['curve'][-1]},{true_nll}"
        )
    (OUTPUT_DIR / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    curves = np.column_stack([samples_seen] + [result["curve"] for result in results.values()])
    header = "samples_seen," + ",".join(results.keys())
    np.savetxt(OUTPUT_DIR / "curves.csv", curves, delimiter=",", header=header, comments="")


def main():
    true_model = build_true_model()
    train_size = 3000
    validation_size = 3000
    batch = 50
    x_val, y_val = collect_sample(true_model, size=validation_size, batch=500, seed=123)
    true_nll = validation_nll(true_model, true_model.eta, x_val, y_val)
    n_steps = len(NaiveBayesEFSampleIterator(true_model, epoch_length=train_size, epochs=1, batch=batch, random_seed=1))
    samples_seen = np.concatenate([np.arange(n_steps) * batch, np.array([train_size])])

    candidates = [(rate, 0.0) for rate in [0.0002, 0.0005, 0.001, 0.002, 0.005]]
    results = {
        "SGD": choose_best_run(SGD_NaiveBayesEF, candidates, true_model, train_size, batch, 7, x_val, y_val),
        "DSNGD": choose_best_run(DSNGD_NaiveBayesEF, candidates, true_model, train_size, batch, 7, x_val, y_val),
    }

    save_curves(samples_seen, results, true_nll)
    save_summary(samples_seen, results, true_nll)

    print(f"Saved results to {OUTPUT_DIR}")
    print(f"True model validation NLL: {true_nll:.6f}")
    for name, result in results.items():
        print(
            f"{name}: lr={result['lr'][0]:g}, initial={result['curve'][0]:.6f}, final={result['curve'][-1]:.6f}"
        )


if __name__ == "__main__":
    main()
