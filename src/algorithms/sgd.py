import numpy as np

from src.model.joint_mlr import JointMLR
from src.algorithms import LineSearch
from tqdm import tqdm


class SGD_JointMLR(LineSearch):
    CLASS_NAME = "SGD"

    def __init__(self, model: JointMLR, name=CLASS_NAME):
        super(SGD_JointMLR, self).__init__(self.gradient_log_conditional_probability, name)
        self.model = model

    def gradient_log_conditional_probability(self, sample, eta):
        x, y = sample
        tx = self.model.T.transform(x)
        q_minus_e = self.model.conditional_probabilities(x, [eta])[0] - self.model.S.identity(y)
        grad_alpha = np.sum(q_minus_e[:, :-1], axis=0)
        grad_beta = np.dot(tx.T, q_minus_e)
        return grad_alpha, grad_beta

    def run(self, sample, starting_point, lr, iter_keep=100, verbose=False, **kwargs):
        alpha, beta = starting_point
        etas = []
        length = max(len(sample) // iter_keep, 1)
        s = tqdm(enumerate(sample), total=len(sample)) if verbose else enumerate(sample)
        for it, obs in s:
            if it % length == 0:
                etas.append([alpha.copy(), beta.copy()])
            g_alpha, g_beta = self.director_process(obs, (alpha, beta))
            r = self.lr_update(it, lr)
            alpha -= r * g_alpha
            beta  -= r * g_beta
        etas.append([alpha.copy(), beta.copy()])
        return etas
