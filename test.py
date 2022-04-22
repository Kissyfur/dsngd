from problems import classification
import numpy as np
import time


def test_kls():
    var = {'Y': 10, 'X': {'discrete': [10, 10, 10, 10], 'gaussian': 0}}
    n = 100
    classi = classification.ClassificationProblem(var)
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

test_kls()
