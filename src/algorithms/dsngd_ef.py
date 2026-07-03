import numpy as np
from tqdm import tqdm

from src.algorithms import LineSearch
from src.model.naive_bayes_ef import NaiveBayesEF


class DSNGD_NaiveBayesEF(LineSearch):
    CLASS_NAME = "DSNGD-EF"

    def __init__(self, model: NaiveBayesEF, name=CLASS_NAME):
        super(DSNGD_NaiveBayesEF, self).__init__(self.approx_natural_gradient_log_conditional_probability, name)
        self.model = model

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

        q_minus_e = self.model.conditional_probabilities(x, eta) - np.eye(self.model.many_classes)[y]
        alpha_full = np.zeros(self.model.many_classes, dtype=float)
        beta_directions = [np.zeros_like(block, dtype=float) for block in self.model.beta_blocks]

        for row, q_row in enumerate(q_minus_e):
            v = np.ones(self.model.many_classes, dtype=float)
            feature_scores = []
            for feature_index, (family, theta_star_block) in enumerate(zip(self.model.families, theta_star_blocks)):
                value = x[row, feature_index]
                scores = np.column_stack(
                    [
                        family.dual_score(value, theta_star_block[:, class_index])
                        for class_index in range(self.model.many_classes)
                    ]
                )
                feature_scores.append(scores)
                v -= np.sum(theta_star_block * scores, axis=0)

            scaled_q = u * q_row
            alpha_full += v * scaled_q
            for beta_direction, scores in zip(beta_directions, feature_scores):
                beta_direction += scores * scaled_q.reshape(1, self.model.many_classes)

        alpha_direction = alpha_full[:-1] - alpha_full[-1]
        return alpha_direction, beta_directions

    def max_entropy_dual_parameter(self, strength=None):
        if strength is None:
            strength = self.model.many_classes
        class_dual = np.ones(self.model.many_classes, dtype=float) * float(strength) / self.model.many_classes
        beta_dual_blocks = []
        for family in self.model.families:
            initial = family.initial_expectation()
            beta_dual_blocks.append(
                np.column_stack(
                    [class_dual[class_index] * initial for class_index in range(self.model.many_classes)]
                )
            )
        return class_dual, beta_dual_blocks

    def update_dual_parameter(self, dual_parameter, sample):
        x, y = sample
        x = self.model._as_feature_matrix(x)
        y = np.asarray(y, dtype=int)
        class_dual, beta_dual_blocks = dual_parameter

        for row, class_index in enumerate(y):
            class_dual[class_index] += 1.0
            for feature_index, (family, block) in enumerate(zip(self.model.families, beta_dual_blocks)):
                block[:, class_index] += family.sufficient_statistic(x[row, feature_index])

    def run(self, sample, starting_point, lr, iter_keep=100, verbose=False, **kwargs):
        alpha, beta_blocks = starting_point
        alpha = alpha.copy()
        beta_blocks = [block.copy() for block in beta_blocks]
        dual_parameter = self.max_entropy_dual_parameter()
        etas = []
        length = max(len(sample) // iter_keep, 1)
        iterator = tqdm(enumerate(sample), total=len(sample)) if verbose else enumerate(sample)

        for it, obs in iterator:
            if it % length == 0:
                etas.append([alpha.copy(), [block.copy() for block in beta_blocks]])
            ng_alpha, ng_beta_blocks = self.director_process(obs, (alpha, beta_blocks), dual_parameter)
            rate = self.lr_update(it, lr)
            alpha -= rate * ng_alpha
            for block, direction in zip(beta_blocks, ng_beta_blocks):
                block -= rate * direction
            self.update_dual_parameter(dual_parameter, obs)

        etas.append([alpha.copy(), [block.copy() for block in beta_blocks]])
        return etas
