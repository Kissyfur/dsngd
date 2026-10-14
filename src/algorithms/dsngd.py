import numpy as np


from src.algorithms import LineSearch
from src.model.joint_mlr import JointMLR
from tqdm import tqdm

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class DSNGD_JointMLR(LineSearch):
    CLASS_NAME = "DSNGD"

    def __init__(self, model: JointMLR, name=CLASS_NAME):
        super(DSNGD_JointMLR, self).__init__(self.aprox_natural_gradient_log_conditional_probability, name)
        self.model = model
        # self.lr_training_function = self.lr_cost_function

    def aprox_natural_gradient_log_conditional_probability(self, sample, eta, dual_parameter):
        x, y = sample
        alpha_dual, beta_dual = dual_parameter
        py_inv = np.sum(alpha_dual) / alpha_dual

        q_minus_e = self.model.conditional_probabilities(x, [eta])[0] - self.model.S.identity(y)

        ng_alpha = np.sum(q_minus_e, axis=0) * py_inv * (1-self.model.T.r)
        ng_beta = None
        for i, (feature_statistic_i, _, dual_rows) in enumerate(self.model.T.slices()):
            xi = x[:, i]
            # beta block of the feature and its term u * q * (1 - <theta*_i, g_i>) of the alpha block
            block, alpha_i = feature_statistic_i.dsngd_terms(xi, beta_dual[dual_rows], alpha_dual, q_minus_e)
            ng_beta = block if ng_beta is None else np.vstack([ng_beta, block])
            ng_alpha += alpha_i

        ng_alpha = ng_alpha[:-1] - ng_alpha[-1]
        return ng_alpha, ng_beta

    def run(self, sample, starting_point, lr, iter_keep=100, verbose=False, **kwargs):
        alpha, beta = starting_point
        # alpha_dual = np.zeros(self.model.S.many_values) + 0.1
        # beta_dual = np.zeros((np.sum(self.model.T.m), self.model.S.many_values)) + 0.1
        alpha_dual, beta_dual = self.max_entropy_dual_parameter()
        etas = []
        length = max(len(sample) // iter_keep, 1)
        s = tqdm(enumerate(sample), total=len(sample)) if verbose else enumerate(sample)
        for it, obs in s:
            if it % length == 0:
                etas.append([alpha.copy(), beta.copy()])
            ng_alpha, ng_beta = self.director_process(obs, (alpha, beta), (alpha_dual, beta_dual))
            r = self.lr_update(it, lr)
            alpha -= r * ng_alpha
            beta -= r * ng_beta
            id_y = self.model.S.identity(obs[1])
            id_x = self.model.T.identity(obs[0])
            cyx = np.dot(id_x.T, id_y)
            alpha_dual += np.sum(id_y, axis=0)
            beta_dual += cyx

        etas.append([alpha.copy(), beta.copy()])
        return etas

    def max_entropy_dual_parameter(self):
        psi = (self.model.T.dim + 1) * self.model.S.many_values
        alpha_dual = np.ones(self.model.S.many_values) * psi / self.model.S.many_values
        beta_dual = np.ones((self.model.T.dual_dim, self.model.S.many_values))
        for feature_statistic_i, _, dual_rows in self.model.T.slices():
            beta_dual[dual_rows, :] = feature_statistic_i.max_entropy_dual(psi, self.model.S.many_values)
        # dual_p = np.random.random((self.cp.y_dimension,self.dp.x_dimension))
        return alpha_dual, beta_dual
