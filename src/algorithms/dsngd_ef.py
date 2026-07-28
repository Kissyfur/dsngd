import numpy as np
from tqdm import tqdm

from src.algorithms import LineSearch
from src.families import CategoricalCoordinate
from src.model.naive_bayes_ef import NaiveBayesEF


class DSNGD_NaiveBayesEF(LineSearch):
    CLASS_NAME = "DSNGD-EF"

    def __init__(self, model: NaiveBayesEF, name=CLASS_NAME):
        super(DSNGD_NaiveBayesEF, self).__init__(self.approx_natural_gradient_log_conditional_probability, name)
        self.model = model
        self._all_categorical = all(isinstance(family, CategoricalCoordinate) for family in self.model.families)

    def approx_natural_gradient_log_conditional_probability(self, sample, eta, dual_parameter):
        x, y = sample
        x = self.model._as_feature_matrix(x)
        y = np.asarray(y, dtype=int)
        class_dual, beta_dual_blocks = dual_parameter
        class_dual = np.asarray(class_dual, dtype=float)
        beta_dual_blocks = [np.asarray(block, dtype=float) for block in beta_dual_blocks]

        if np.any(class_dual <= 0.0):
            raise ValueError("class dual parameters must be positive")

        u = np.sum(class_dual) / class_dual
        theta_star_blocks = [
            block / class_dual.reshape(1, self.model.many_classes)
            for block in beta_dual_blocks
        ]

        q_minus_e = self.model.conditional_probabilities(x, eta)
        q_minus_e[np.arange(len(y)), y] -= 1.0
        if self._all_categorical:
            return self._categorical_direction(x, q_minus_e, class_dual, beta_dual_blocks)

        v = np.ones_like(q_minus_e)
        feature_scores = []
        for feature_index, (family, theta_star_block) in enumerate(zip(self.model.families, theta_star_blocks)):
            scores = family.dual_score(x[:, feature_index], theta_star_block.T)
            feature_scores.append(scores)
            v -= np.einsum("cd,ncd->nc", theta_star_block.T, scores)

        scaled_q = q_minus_e * u
        alpha_full = np.sum(v * scaled_q, axis=0)
        beta_directions = [
            np.einsum("ncd,nc->dc", scores, scaled_q)
            for scores in feature_scores
        ]

        alpha_direction = alpha_full[:-1] - alpha_full[-1]
        return alpha_direction, beta_directions

    def _categorical_direction(self, x, q_minus_e, class_dual, beta_dual_blocks):
        total_dual = np.sum(class_dual)
        u = total_dual / class_dual
        alpha_full = np.sum(q_minus_e * u, axis=0) * (1.0 - len(self.model.families))
        beta_directions = []

        for feature_index, (family, block) in enumerate(zip(self.model.families, beta_dual_blocks)):
            xi = x[:, feature_index].astype(int)
            if np.any((xi < 0) | (xi >= family.many_values)):
                raise ValueError("categorical values must be in [0, many_values)")

            probabilities = np.empty((family.many_values, self.model.many_classes), dtype=float)
            probabilities[:-1] = block / class_dual.reshape(1, -1)
            probabilities[-1] = 1.0 - np.sum(probabilities[:-1], axis=0)
            if np.any(probabilities <= family.min_probability):
                raise ValueError("expectation_parameter must be in the categorical simplex interior")

            d_q = q_minus_e * (u / probabilities[xi])
            baseline_rows = xi == family.dim
            beta_direction = family.sufficient_statistic(xi).T @ d_q
            if np.any(baseline_rows):
                baseline_sum = np.sum(d_q[baseline_rows], axis=0)
                alpha_full += baseline_sum
                beta_direction -= baseline_sum.reshape(1, -1)
            beta_directions.append(beta_direction)

        alpha_direction = alpha_full[:-1] - alpha_full[-1]
        return alpha_direction, beta_directions

    def max_entropy_dual_parameter(self, strength=None):
        if strength is None:
            strength = self.model.many_classes
        class_dual = np.ones(self.model.many_classes, dtype=float) * float(strength) / self.model.many_classes
        beta_dual_blocks = []
        for family in self.model.families:
            initial = family.initial_expectation()
            beta_dual_blocks.append(initial.reshape(-1, 1) * class_dual.reshape(1, -1))
        return class_dual, beta_dual_blocks

    def update_dual_parameter(self, dual_parameter, sample):
        x, y = sample
        x = self.model._as_feature_matrix(x)
        y = np.asarray(y, dtype=int)
        class_dual, beta_dual_blocks = dual_parameter

        class_dual += np.bincount(y, minlength=self.model.many_classes)
        if self._all_categorical:
            for feature_index, (family, block) in enumerate(zip(self.model.families, beta_dual_blocks)):
                xi = x[:, feature_index].astype(int)
                non_baseline = xi < family.dim
                np.add.at(block, (xi[non_baseline], y[non_baseline]), 1.0)
            return

        class_indicators = np.eye(self.model.many_classes)[y]
        for feature_index, (family, block) in enumerate(zip(self.model.families, beta_dual_blocks)):
            statistics = family.sufficient_statistic(x[:, feature_index])
            block += statistics.T @ class_indicators

    def run(self, sample, starting_point, lr, iter_keep=100, verbose=False, **kwargs):
        alpha, beta_blocks = starting_point
        alpha = alpha.copy()
        beta_blocks = [block.copy() for block in beta_blocks]
        dual_parameter = self.max_entropy_dual_parameter()
        etas = []
        length = max(len(sample) // iter_keep, 1)
        desc = kwargs.get("desc", self.key)
        iterator = tqdm(enumerate(sample), total=len(sample), desc=desc) if verbose else enumerate(sample)

        for it, obs in iterator:
            if it % length == 0:
                etas.append([alpha.copy(), [block.copy() for block in beta_blocks]])
            ng_alpha, ng_beta_blocks = self.director_process(obs, (alpha, beta_blocks), dual_parameter)
            rate = self.lr_update(it, lr)
            alpha -= rate * ng_alpha
            for block, direction in zip(beta_blocks, ng_beta_blocks):
                block -= rate * direction
            alpha, beta_blocks = self.model.project_eta((alpha, beta_blocks))
            self.update_dual_parameter(dual_parameter, obs)

        etas.append([alpha.copy(), [block.copy() for block in beta_blocks]])
        return etas
