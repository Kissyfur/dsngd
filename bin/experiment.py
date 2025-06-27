import logging
import numpy as np

from src.model.joint_mlr import JointMLR
from src.data.sample_creator import JointMLRSampleIterator
from tqdm import tqdm
from src.algorithms.sgd import SGD_JointMLR
from src.algorithms.dsngd import DSNGD_JointMLR
from src.grapher import plot_lines
from src.algorithms.sngd import SNGD_JointMLR
from src.algorithms.adaGrad import AdaGrad_JointMLR


logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    n = 10000000
    epochs = 1
    batch = 250
    n_lr = 500 * batch
    many_experiments = 1
    entropies = [0.1, 0.7, 1]
    # entropies = [1]
    M_1 = (10, [10, 5])
    M_2 = (20, [10, 5, 10, 5])
    M_3 = (30, [10, 5, 10, 5, 10, 5])
    manifolds = [M_1, M_2, M_3]
    algorithm_classes = [SGD_JointMLR, AdaGrad_JointMLR, DSNGD_JointMLR, SNGD_JointMLR]
    algorithm_classes = [SGD_JointMLR, AdaGrad_JointMLR, DSNGD_JointMLR]
    # algorithm_classes = [DSNGD_JointMLR]
    many_algs = len(algorithm_classes)
    d = {}
    gr = np.zeros((3, 3, many_algs, 101))
    up = np.zeros((3, 3, many_algs, 101))
    dn = np.zeros((3, 3, many_algs, 101))
    problems = {}
    for i, (s, m) in enumerate(manifolds):
        for j, sigma_sq in enumerate(entropies):
            d = {algo_class.CLASS_NAME: [] for algo_class in algorithm_classes}
            for exp_num in range(many_experiments):
                logging.info(f"Running experiment")
                logging.info(f"Manifold: {(s, m)},     Entropy: {sigma_sq},      Experiment num: {exp_num}")
                problem = JointMLR(s, m)
                np.random.seed(exp_num)
                problem.set_random_eta(sigma_sq)
                problems[(i, j, exp_num)] = problem
                sample = JointMLRSampleIterator(problem, n, epochs, batch, random_seed=exp_num)
                sample_lr = JointMLRSampleIterator(problem, n_lr, 1, batch, random_seed=exp_num+1)
                H = []
                for algo_class in algorithm_classes:
                    logging.info(f"Algorithm: {algo_class.CLASS_NAME}")
                    model = JointMLR(s, m)
                    algo = algo_class(model)
                    # lr = model.find_best_lr(problem, sample_lr, algo) if algo.CLASS_NAME != 'DSNGD' else  np.array([0.0001, 0.01])
                    lr = model.find_best_lr(problem, sample_lr, algo)
                    model.fit(sample, algo, verbose=True, lr=lr)
                    model.compute_history_metrics(problem)
                    d[algo.CLASS_NAME].append(model.history['error'])
            gr[i, j] = np.array([np.median(d[algo_class.CLASS_NAME], axis=0)
                                  for algo_class in algorithm_classes])
            up[i, j] = np.array([np.percentile(d[algo_class.CLASS_NAME], q=75, axis=0)
                                  for algo_class in algorithm_classes])
            dn[i, j] = np.array([np.percentile(d[algo_class.CLASS_NAME], q=25, axis=0)
                                  for algo_class in algorithm_classes])
    labels = [algo_class.CLASS_NAME for algo_class in algorithm_classes]
    x = np.arange(101) * n*epochs // 101
    plot_lines(x, gr, labels, y_labels=['M1', 'M2', 'M3'], x_labels=['High entropy', 'Medium entropy', 'Low entropy'], low_lines=up, high_lines=dn)




    print("finished")
