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

    def aprox_natural_gradient_log_conditional_probability(self, sample, eta, dual_parameter, identity=None,
                                                          points=None):
        # Theorem 3.2 summed over the mini-batch (Remark 4.12). The rescaled residuals w = D_Y(zeta*) q, their sum a
        # and the sums M of identity(x) weighted by w, which hold T(x)^T w and the weighted counts of all the values of
        # the categorical features, come from one product for all the features. Each family computes the blocks
        # H(theta*) (M - a theta*) of its features together from their auxiliary point, and c = a - sum of
        # <theta*, block>. identity is the statistic of x accumulated by the dual parameter, if already computed, and
        # points the auxiliary points of the groups of features, computed from the dual parameter if not given.
        x, y = sample
        alpha_dual = dual_parameter[0]
        identity = self.model.T.identity(x) if identity is None else identity
        points = self.auxiliary_points(dual_parameter) if points is None else points
        py_inv = np.sum(alpha_dual) / alpha_dual

        tx = identity[:, self.model.T.expectation_rows]
        w = (self.model.conditional_probabilities(x, [eta], tx)[0] - self.model.S.identity(y)) * py_inv
        a = np.sum(w, axis=0)
        m = np.dot(identity.T, w)
        ng_alpha = a.copy()
        ng_beta = np.empty((self.model.T.dim, len(a)))
        for group, point in zip(self.model.T.groups, points):
            block, products = group.statistic.dsngd_block(point, group.stack(m[group.dual_rows]), a)
            ng_beta[group.rows] = group.unstack(block)
            ng_alpha -= products

        ng_alpha = ng_alpha[:-1] - ng_alpha[-1]
        return ng_alpha, ng_beta

    def auxiliary_points(self, dual_parameter, previous=None):
        # Auxiliary point of each group of features from the dual parameter. The families that keep it in natural
        # coordinates follow the dual parameter from previous, the points of the previous iteration.
        alpha_dual, beta_dual = dual_parameter
        previous = [None] * len(self.model.T.groups) if previous is None else previous
        return [group.statistic.auxiliary_point(group.stack(beta_dual[group.dual_rows]), alpha_dual, point)
                for group, point in zip(self.model.T.groups, previous)]

    def run(self, sample, starting_point, lr, iter_keep=100, verbose=False, **kwargs):
        alpha, beta = starting_point
        # alpha_dual = np.zeros(self.model.S.many_values) + 0.1
        # beta_dual = np.zeros((np.sum(self.model.T.m), self.model.S.many_values)) + 0.1
        alpha_dual, beta_dual = self.max_entropy_dual_parameter()
        points = self.auxiliary_points((alpha_dual, beta_dual))
        etas = []
        length = max(len(sample) // iter_keep, 1)
        s = tqdm(enumerate(sample), total=len(sample)) if verbose else enumerate(sample)
        for it, obs in s:
            if it % length == 0:
                etas.append([alpha.copy(), beta.copy()])
            # The statistic of the batch serves the direction and the update of the dual parameter
            id_x = self.model.T.identity(obs[0])
            ng_alpha, ng_beta = self.director_process(obs, (alpha, beta), (alpha_dual, beta_dual), id_x, points)
            r = self.lr_update(it, lr)
            alpha -= r * ng_alpha
            beta -= r * ng_beta
            id_y = self.model.S.identity(obs[1])
            cyx = np.dot(id_x.T, id_y)
            alpha_dual += np.sum(id_y, axis=0)
            beta_dual += cyx
            points = self.auxiliary_points((alpha_dual, beta_dual), points)

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
