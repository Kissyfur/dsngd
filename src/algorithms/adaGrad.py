from src.algorithms.sgd import SGD_JointMLR
from src.model.joint_mlr import JointMLR
import numpy as np
from tqdm import tqdm

class AdaGrad_JointMLR(SGD_JointMLR):
    CLASS_NAME = "AdaGrad"

    def __init__(self, model: JointMLR, name=CLASS_NAME):
        super(AdaGrad_JointMLR, self).__init__(model, name)
        self.single_learning_rate_parameter = True
        self.lr_update = self.adagrad_learning_rate

    @staticmethod
    def adagrad_learning_rate(gti, learning_rate_param):
        fudge_factor = 1e-8
        alpha_lr = learning_rate_param / np.sqrt(fudge_factor + gti[0])
        beta_lr  = learning_rate_param / np.sqrt(fudge_factor + gti[1])
        return alpha_lr, beta_lr

    def run(self, sample, starting_point, lr, verbose=False, iter_keep=100, **kwargs):
        alpha, beta = starting_point
        etas = []
        length = max(len(sample) // iter_keep, 1)
        Gt = [0, 0]
        s = tqdm(enumerate(sample), total=len(sample)) if verbose else enumerate(sample)
        for it, obs in s:
            if it % length == 0:
                etas.append([alpha.copy(), beta.copy()])
            g_alpha, g_beta = self.director_process(obs, (alpha, beta))
            Gt[0] += g_alpha * g_alpha
            Gt[1] += g_beta * g_beta
            lr_alpha, lr_beta = self.lr_update(Gt, lr)
            alpha -= lr_alpha * g_alpha
            beta  -= lr_beta * g_beta
        etas.append([alpha.copy(), beta.copy()])
        return etas