from src.algorithms.dsngd import DSNGD_JointMLR
from src.model.joint_mlr import JointMLR
from tqdm import tqdm


class SNGD_JointMLR(DSNGD_JointMLR):
    CLASS_NAME = 'SNGD'

    def __init__(self,  model: JointMLR, name=CLASS_NAME):
        super(SNGD_JointMLR, self).__init__(model, name)

    def run(self, sample, starting_point, lr, iter_keep=100, verbose=False, **kwargs):
        alpha, beta = starting_point
        alpha_dual, beta_dual = self.model.to_dual([alpha, beta])
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
            alpha_dual, beta_dual = self.model.to_dual([alpha, beta])
        etas.append([alpha.copy(), beta.copy()])
        return etas

