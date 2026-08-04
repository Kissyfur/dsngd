import numpy as np
from tqdm import tqdm

from src.algorithms import log_spaced_checkpoint_iterations
from src.algorithms.sgd_ef import SGD_NaiveBayesEF
from src.model.naive_bayes_ef import NaiveBayesEF


class AdaGrad_NaiveBayesEF(SGD_NaiveBayesEF):
    CLASS_NAME = "AdaGrad-EF"

    def __init__(self, model: NaiveBayesEF, name=CLASS_NAME):
        super(AdaGrad_NaiveBayesEF, self).__init__(model, name)
        self.single_learning_rate_parameter = True
        self.lr_update = self.adagrad_learning_rate

    @staticmethod
    def adagrad_learning_rate(gradient_squares, learning_rate_param):
        fudge_factor = 1e-8
        rate = float(np.asarray(learning_rate_param).reshape(-1)[0])
        alpha_rate = rate / np.sqrt(fudge_factor + gradient_squares[0])
        beta_rates = [
            rate / np.sqrt(fudge_factor + block_squares)
            for block_squares in gradient_squares[1]
        ]
        return alpha_rate, beta_rates

    def run(self, sample, starting_point, lr, iter_keep=100, verbose=False, **kwargs):
        alpha, beta_blocks = starting_point
        alpha = alpha.copy()
        beta_blocks = [block.copy() for block in beta_blocks]
        gradient_squares = [
            np.zeros_like(alpha),
            [np.zeros_like(block) for block in beta_blocks],
        ]
        etas = []
        checkpoints = set(log_spaced_checkpoint_iterations(len(sample), iter_keep))
        desc = kwargs.get("desc", self.key)
        iterator = tqdm(enumerate(sample), total=len(sample), desc=desc) if verbose else enumerate(sample)

        for it, obs in iterator:
            if it in checkpoints:
                etas.append([alpha.copy(), [block.copy() for block in beta_blocks]])
            grad_alpha, grad_beta_blocks = self.director_process(obs, (alpha, beta_blocks))
            gradient_squares[0] += grad_alpha * grad_alpha
            for block_squares, gradient in zip(gradient_squares[1], grad_beta_blocks):
                block_squares += gradient * gradient
            alpha_rate, beta_rates = self.lr_update(gradient_squares, lr)
            alpha -= alpha_rate * grad_alpha
            for block, rate, gradient in zip(beta_blocks, beta_rates, grad_beta_blocks):
                block -= rate * gradient
            alpha, beta_blocks = self.model.project_eta((alpha, beta_blocks))

        etas.append([alpha.copy(), [block.copy() for block in beta_blocks]])
        return etas
