

class Metrics_JointMLR:
    def __init__(self):
        return





    def minus_conditional_log_likelihood_to_sample(self, sample):
        def minus_cll(parameters):
            return -self.cll(self.cp.log_p, sample, parameters)

        return minus_cll

    def cll(self, log_p, sample, param):
        e = self.evaluate_per_param(log_p, sample, param)
        return e

    def evaluate_per_param(self, f, sample, params):
        n = len(sample)
        B = 100
        val = 0.
        for batch in range(0, n, B):
            s = sample[batch:batch + B]
            val += np.sum(f(s, params), axis=1)
        return val  # /n