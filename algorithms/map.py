import numpy as np
from algorithms.algorithm import Algorithm


# Discrete case
class MAP(Algorithm):
    def __init__(self, problem):
        super(MAP, self).__init__()
        self.key = 'MAP'
        self.y_values = problem.y_values
        self.run_algorithm = self.mAp
        self.single_learning_rate_parameter = False

    def mAp(self, data):
        sample = data['sample']
        start = data['start']
        alpha = data['alpha']

        total_iter = data['total iterations']
        self.set_run(total_iter)
        parameters = []
        current_parameter = start.copy() * alpha
        i = 0
        for obs in sample:
            if self.save_current_parameter(i):
                parameters.append(current_parameter.copy())
            # current_parameter += np.sum(map(np.kron, self.ty(obs[:, 0]), obs[:, 1:]), axis=0)
            current_parameter += np.dot(self.ty(obs[:, 0]).T, obs[:, 1:])
            i = i+1
        parameters.append(current_parameter.copy())
        return np.array(parameters)
        
    def ty(self, y):
        y = y.copy().astype(int)
        id_y = np.identity(self.y_values, dtype=int)
        ty = id_y[y]
        return ty

