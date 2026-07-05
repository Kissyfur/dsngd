import numpy as np
from tqdm import tqdm

from src.algorithms import LineSearch
from src.model.naive_bayes_ef import NaiveBayesEF


class SGD_NaiveBayesEF(LineSearch):
    CLASS_NAME = "SGD-EF"

    def __init__(self, model: NaiveBayesEF, name=CLASS_NAME):
        super(SGD_NaiveBayesEF, self).__init__(self.gradient_log_conditional_probability, name)
        self.model = model

    def gradient_log_conditional_probability(self, sample, eta):
        x, y = sample
        x = self.model._as_feature_matrix(x)
        y = np.asarray(y, dtype=int)
        q_minus_e = self.model.conditional_probabilities(x, eta) - np.eye(self.model.many_classes)[y]

        grad_alpha = np.sum(q_minus_e[:, :-1], axis=0)
        grad_beta_blocks = [np.zeros_like(block, dtype=float) for block in self.model.beta_blocks]

        for feature_index, (family, grad_block) in enumerate(zip(self.model.families, grad_beta_blocks)):
            statistics = family.sufficient_statistic(x[:, feature_index])
            grad_block += statistics.T @ q_minus_e

        return grad_alpha, grad_beta_blocks

    def run(self, sample, starting_point, lr, iter_keep=100, verbose=False, **kwargs):
        alpha, beta_blocks = starting_point
        alpha = alpha.copy()
        beta_blocks = [block.copy() for block in beta_blocks]
        etas = []
        length = max(len(sample) // iter_keep, 1)
        desc = kwargs.get("desc", self.key)
        iterator = tqdm(enumerate(sample), total=len(sample), desc=desc) if verbose else enumerate(sample)

        for it, obs in iterator:
            if it % length == 0:
                etas.append([alpha.copy(), [block.copy() for block in beta_blocks]])
            grad_alpha, grad_beta_blocks = self.director_process(obs, (alpha, beta_blocks))
            rate = self.lr_update(it, lr)
            alpha -= rate * grad_alpha
            for block, gradient in zip(beta_blocks, grad_beta_blocks):
                block -= rate * gradient
            alpha, beta_blocks = self.model.project_eta((alpha, beta_blocks))

        etas.append([alpha.copy(), [block.copy() for block in beta_blocks]])
        return etas
