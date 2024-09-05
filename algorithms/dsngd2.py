from algorithms.dsngd import DSNGD


class DSNGD2(DSNGD):
    def __init__(self, natural_gradient_aproximation, dual_estimator, batch=1):
        super(DSNGD2, self).__init__(natural_gradient_aproximation, dual_estimator, batch)
        self.key = 'dsngd2'
