import numpy as np
from scipy.special import logsumexp


class NaiveBayesEF:
    """Naive Bayes model whose feature coordinates are exponential families."""

    def __init__(self, many_classes, families, name="model"):
        if many_classes < 2:
            raise ValueError("many_classes must be at least 2")
        if not families:
            raise ValueError("at least one feature family is required")

        self.many_classes = int(many_classes)
        self.families = list(families)
        self.name = name
        self.alpha = np.zeros(self.many_classes - 1, dtype=float)
        self.beta_blocks = [
            np.repeat(
                family.natural_from_expectation(family.initial_expectation()).reshape(-1, 1),
                self.many_classes,
                axis=1,
            )
            for family in self.families
        ]
        self.eta = (self.alpha, self.beta_blocks)

    @property
    def feature_dim(self):
        return sum(family.dim for family in self.families)

    @property
    def parameter_dim(self):
        return self.alpha.size + self.many_classes * self.feature_dim

    def set_eta(self, eta):
        if not self.valid_eta(eta):
            raise ValueError("eta has incompatible shapes")
        alpha, beta_blocks = eta
        self.alpha = np.asarray(alpha, dtype=float)
        self.beta_blocks = [np.asarray(block, dtype=float) for block in beta_blocks]
        self.eta = (self.alpha, self.beta_blocks)

    def valid_eta(self, eta):
        if len(eta) != 2:
            return False
        alpha, beta_blocks = eta
        if np.asarray(alpha).shape != (self.many_classes - 1,):
            return False
        if len(beta_blocks) != len(self.families):
            return False
        for family, block in zip(self.families, beta_blocks):
            if np.asarray(block).shape != (family.dim, self.many_classes):
                return False
        return True

    def project_eta(self, eta):
        if not self.valid_eta(eta):
            raise ValueError("eta has incompatible shapes")
        alpha, beta_blocks = eta
        return (
            np.asarray(alpha, dtype=float),
            [
                family.project_natural(np.asarray(block, dtype=float))
                for family, block in zip(self.families, beta_blocks)
            ],
        )

    def class_log_weights(self, eta=None):
        alpha, beta_blocks = self._resolve_eta(eta)
        alpha_s = np.concatenate([alpha, np.zeros(1, dtype=float)])
        log_weights = alpha_s.copy()
        for family, block in zip(self.families, beta_blocks):
            log_weights += family.log_partition(block.T)
        return log_weights

    def log_class_probabilities(self, eta=None):
        log_weights = self.class_log_weights(eta)
        return log_weights - logsumexp(log_weights)

    def class_probabilities(self, eta=None):
        return np.exp(self.log_class_probabilities(eta))

    def log_conditional_probabilities(self, x, eta=None):
        x = self._as_feature_matrix(x)
        _, beta_blocks = self._resolve_eta(eta)
        log_probabilities = np.tile(self.log_class_probabilities(eta), (len(x), 1))
        for feature_index, (family, block) in enumerate(zip(self.families, beta_blocks)):
            values = x[:, feature_index:feature_index + 1]
            log_probabilities += family.log_density(values, block.T)
        normalizer = logsumexp(log_probabilities, axis=1, keepdims=True)
        return log_probabilities - normalizer

    def conditional_probabilities(self, x, eta=None):
        return np.exp(self.log_conditional_probabilities(x, eta))

    def to_dual(self, eta=None):
        class_probabilities = self.class_probabilities(eta)
        _, beta_blocks = self._resolve_eta(eta)
        beta_dual_blocks = []
        for family, block in zip(self.families, beta_blocks):
            expectations = family.expectation_from_natural(block.T).T
            beta_dual_blocks.append(expectations * class_probabilities.reshape(1, -1))
        return class_probabilities, beta_dual_blocks

    def _resolve_eta(self, eta):
        if eta is None:
            return self.alpha, self.beta_blocks
        if not self.valid_eta(eta):
            raise ValueError("eta has incompatible shapes")
        alpha, beta_blocks = eta
        return np.asarray(alpha, dtype=float), [np.asarray(block, dtype=float) for block in beta_blocks]

    def _as_feature_matrix(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != len(self.families):
            raise ValueError(f"expected feature matrix with {len(self.families)} columns")
        return x
