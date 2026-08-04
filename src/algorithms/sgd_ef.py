import numpy as np
from tqdm import tqdm

from src.algorithms import LineSearch, log_spaced_checkpoint_iterations
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
        q_minus_e = self.model.conditional_probabilities(x, eta)
        q_minus_e[np.arange(len(y)), y] -= 1.0

        grad_alpha = np.sum(q_minus_e[:, :-1], axis=0)
        grad_beta_blocks = [
            family.sufficient_statistic(observations).T @ q_minus_e
            for family, observations in self.model.family_observations(x)
        ]
        return grad_alpha, grad_beta_blocks

    def run(self, sample, starting_point, lr, iter_keep=100, verbose=False, **kwargs):
        alpha, beta_blocks = starting_point
        alpha = alpha.copy()
        beta_blocks = [block.copy() for block in beta_blocks]
        etas = []
        checkpoints = set(log_spaced_checkpoint_iterations(len(sample), iter_keep))
        desc = kwargs.get("desc", self.key)
        iterator = tqdm(enumerate(sample), total=len(sample), desc=desc) if verbose else enumerate(sample)

        for it, obs in iterator:
            if it in checkpoints:
                etas.append([alpha.copy(), [block.copy() for block in beta_blocks]])
            grad_alpha, grad_beta_blocks = self.director_process(obs, (alpha, beta_blocks))
            rate = self.lr_update(it, lr)
            alpha -= rate * grad_alpha
            for block, gradient in zip(beta_blocks, grad_beta_blocks):
                block -= rate * gradient
            alpha, beta_blocks = self.model.project_eta((alpha, beta_blocks))

        etas.append([alpha.copy(), [block.copy() for block in beta_blocks]])
        return etas
