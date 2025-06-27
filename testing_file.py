import numpy as np
import time
from src.model.joint_mlr import JointMLR
from src.algorithms.sgd import SGD_JointMLR
from src.algorithms.dsngd import DSNGD_JointMLR
from src.algorithms.adaGrad import AdaGrad_JointMLR
from src.data.sample_creator import JointMLRSampleIterator


def trying():
    s, m = 30, [5, 10, 5, 10, 5, 10]
    n = 10000000
    problem = JointMLR(s, m)
    model = JointMLR(s, m)
    np.random.seed(0)
    problem.set_random_eta(1)
    # alpha = np.array([-1, 0.5])
    # beta = np.array([[ 1, -1.1, -0.6,  0.9],
    #                  [1.7, -0.7,  0.3,  -0.2],
    #                  [-2.0, -0.3,  -0.3,  1.1]]).T
    # problem.set_eta([alpha, beta])

    sample = JointMLRSampleIterator(problem, n, 1, 250, random_seed=0)
    # x, y = np.array([[1, 0]]), np.array([2])

    # sample = [[np.array([[0, 3], [1, 0]]), np.array([1, 2])]]*50

    dsngd = DSNGD_JointMLR(model)
    sgd = SGD_JointMLR(model)
    adagrad = AdaGrad_JointMLR(model)
    model.fit(sample, dsngd, verbose=True, lr=[0.0001, 0.01])

    # dsngd.run(sample, [model.alpha, model.beta], lr=[1., 1.])

    model.compute_history_metrics(problem)

    # print(model.history['etas'])

    print(model.history['error'])

    # q_minus_e = np.squeeze(problem.conditional_probabilities(x, [[alpha, beta]])) - problem.S.identity(y)
    # print(q_minus_e)

def test_kls():
    var = {'Y': 10, 'X': {'discrete': [10, 10, 10, 10], 'gaussian': 0}}
    n = 100
    classi = joint_mlr_problem.ClassificationProblem(var)
    parameter = classi.set_problem(1)
    classi.cp.set_pyx(parameter)
    dim = (n, classi.cp.y_dimension, classi.cp.x_dimension)
    params = np.random.normal(0., 1., size=dim)

    t1 = time.time()
    kl1_f = classi.kl_to_real_parameter(parameter)
    kl1 = kl1_f(params)

    t2 = time.time()
    kl2_f = classi.cp.kl_divergence
    kl2 = kl2_f(params)
    t3 = time.time()
    print("Method 1 needed: % seconds" %(t2-t1))
    print("Method 2 needed: % seconds" %(t3-t2))
    print("Error between two methods is: ", np.sum(np.abs(kl1-kl2)))
    print("kl to solution: %f" %kl2_f(parameter))

if __name__ == "__main__":
    # test_kls()
    trying()

