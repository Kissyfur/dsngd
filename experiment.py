import numpy as np
import os

from scipy.special import logsumexp
from problems import classification
from problems import utils
import grapher as gr

from algorithms import sgd
from algorithms import dsngd
from algorithms import dsngd2
from algorithms import adaGrad
from algorithms import adadelta

# matplotlib.use('TkAgg')

current_path = os.getcwd()
directory = '/saved_data/'
path = current_path + directory
try:
    os.mkdir(path)
except OSError:
    print("Creation of the directory %s failed" % path, ". Maybe already exists.")
else:
    print("Successfully created the directory %s " % path)


def experiment(problem, algs, n, many_experiments=2, epochs=1, batch=1, extremality=1.):
    sample_from_real_parameter = True
    lr_iterations = 500
    data = {'start': problem.start['sgd'], 'dual_start': problem.dual_start['dsngd'],
            'total iterations': n * epochs // batch, 'total lr iterations': lr_iterations}
    m_vals = {}
    for alg in algs:
        m_vals[alg.key] = []
    for i in range(many_experiments):
        real_parameter = problem.set_problem(extremality, seed=i)
        print("\nExperiment num: ", i, ". Real beta has dim:", problem.cp.dimension)

        # Create sample and cost fucntion
        print("Creating new sample...")
        data['sample_creator'] = None
        data['sample_big_creator'] = None
        cost_function = None
        if sample_from_real_parameter:
            problem.cp.set_pyx(real_parameter)
            sample_big = problem.generate_sample_big(n, epochs=epochs, batch=batch)
            sample = problem.generate_sample(n, epochs=epochs, batch=batch)
            sample_big_lr = problem.generate_sample_big(lr_iterations * batch, epochs=1, batch=batch,
                                                        seed=10)
            sample_lr = problem.generate_sample(lr_iterations * batch, epochs=1, batch=batch, seed=10)
            data['sample_creator'] = sample
            data['sample_big_creator'] = sample_big
            data['sample_lr_creator'] = sample_lr
            data['sample_big_lr_creator'] = sample_big_lr
            # kl = problem.kl_to_real_parameter(real_parameter)
            kl = problem.cp.kl_divergence
            cost_function = kl
            data["cost_function"] = cost_function
        else:
            print("Random sample iterator not implemented yet. Do it!")
            # cll = problem.minus_conditional_log_likelihood_to_sample(sample_cp)
            # cost_function = cll
            exit()

        optimum = cost_function(real_parameter)
        for alg in algs:
            fn = path + str(i) + alg.key + str(extremality) + problem.get_file_name() + str(n) + str(batch) + str(epochs)
            # fn = ''
            err = optimize_function(alg, data, file_name=fn)
            print(np.min(err), optimum)
            best = np.min([np.min(err), optimum])
            m_vals[alg.key].append(err - best)
    return m_vals


def graph_3_dims_3_entropies(sample_length, many_experiments=100, epochs=1, batch=1):
    var1 = {'Y': 10, 'X': {'discrete': [10, 5], 'gaussian': 0}}
    var2 = {'Y': 20, 'X': {'discrete': [10, 5, 10, 5], 'gaussian': 0}}
    var3 = {'Y': 30, 'X': {'discrete': [10, 5, 10, 5, 10, 5], 'gaussian': 0}}
    dims = [var1, var2, var3]

    extremalities = [0.1, 0.5, 1.]
    medians = [[0,0,0],[0,0,0],[0,0,0]]
    first_quartile = [[0,0,0],[0,0,0],[0,0,0]]
    third_quartile = [[0,0,0],[0,0,0],[0,0,0]]

    x_labels = ['High entropy', 'Medium entropy', 'Low entropy']
    labels = []
    y_labels = []
    n_dots = 1
    for i in range(len(dims)):
        dim = dims[i]
        classi = classification.ClassificationProblem(dim)
        y_labels.append(str(classi.cp.dimension - 1) + '      ')
        SGD = sgd.SGD(gradient=classi.direction['sgd'])
        # adagrad = adaGrad.AdaGrad(gradient=classi.direction['sgd'])
        # Adadelta = adadelta.Adadelta(gradient=classi.direction['sgd'])
        DSNGD = dsngd.DSNGD(classi.direction['dsngd'], classi.dual_estimator['dsngd'])
        # dsngd2 = dsngd2.DSNGD2(classi.direction['dsngd2'], classi.dual_estimator['dsngd'])
        algs = [SGD, DSNGD]
        for j in range(len(extremalities)):
            extremality = extremalities[j]
            e = experiment(classi, algs, n=sample_length, many_experiments=many_experiments,
                           epochs=epochs, batch=batch, extremality=extremality)
            lines = []
            labels = []
            for key in e:
                labels.append(key)
                lines.append(e[key])
            medians[i][j] = np.median(lines, axis=1)
            first_quartile[i][j] = np.percentile(lines, 75, axis=1)
            third_quartile[i][j] = np.percentile(lines, 25, axis=1)
            n_dots = len(medians[i][j][0])
    medians = np.array(medians)
    first_quartile = np.array(first_quartile)
    third_quartile = np.array(third_quartile)
    x = np.arange(n_dots) * sample_length // n_dots

    file_name = '9experiments'
    gr.plot_lines(x, medians, labels, x_labels=x_labels, y_labels=y_labels,
                  low_lines=first_quartile, high_lines=third_quartile, file_name=file_name)


def run_algorithm(alg, data, file=''):
    estimations = None
    try:
        estimations = np.load(file + '.npy')
        return estimations
    except Exception as e:
        print("ERROR : " + str(e))
    # At this point, estimations is still None
    print("Running the experiment. Adjusting hyperparameters of ", alg.key)
    lr = alg.adjust_lr_with_data(data, progress_bar=True)
    data['lr'] = lr
    try:
        print("Running ", alg.key, "...")
        data['sample'] = data['sample_creator']()
        data['sample_big'] = data['sample_big_creator']()
        estimations = alg.run_algorithm(data, progress_bar=True)
        print("Done")
    except (OverflowError, np.linalg.linalg.LinAlgError, FloatingPointError):
        print("The best Learning rate failed to converge in whole sample")
        exit()
    if file:
        try:
            np.save(file, estimations)
        except Exception as e:
            print("ERROR : " + str(e))
            print("Couldn't save the estimations found by the algroithm")
    return estimations


def optimize_function(algorithm, data, file_name=''):
    try:
        errors = np.load(file_name + '.cost' + '.npy')
        return errors
    except Exception as e:
        print("ERROR : " + str(e))
    # Errors is not loaded. Compute them.
    estimations = run_algorithm(algorithm, data, file_name)
    print("Computing errors of ", algorithm.key, " estimates...")
    cost_function = data["cost_function"]
    errors = cost_function(estimations)
    print("Done")
    if file_name:
        try:
            np.save(file_name + '.cost', errors)
        except Exception as e:
            print("ERROR : " + str(e))
            print("Couldn't save the errors found by the algorithm")
    return errors

def test_kl_and_fast_kl():
    var = {'Y': 10, 'X': {'discrete': [3, 5, 9, 10, 5], 'gaussian': 0}}
    sample_from_real_parameter = True

    # Create problem
    classi = classification.ClassificationProblem(var)
    real_parameter = classi.set_problem()
    X = classi.cp.compute_all_x()
    TX = classi.cp.t_x(X)
    log_p_real = classi.cp.compute_log_numerators(TX, real_parameter)
    log_p_real = log_p_real - logsumexp(log_p_real)
    p_real = np.exp(log_p_real)

    def ckl_to_real_parameter(params):
        log_q = classi.cp.all_log_p(TX, params)
        qs = np.exp(log_q)
        return utils.kl_divergence(p_real[0], qs)
    dim = (100, classi.cp.y_dimension, classi.cp.x_dimension)
    np.random.seed(1)
    params = np.random.random(size=dim)
    kl = ckl_to_real_parameter(params)


if __name__ == "__main__":
    # var = {'Y': 3, 'X': {'discrete': [6, 5], 'gaussian': 0}}
    # classi = classification.ClassificationProblem(var)
    # sgd = sgd.SGD(gradient=classi.direction['sgd'])
    # adagrad = adaGrad.AdaGrad(gradient=classi.direction['sgd'])
    # adadelta = adadelta.Adadelta(gradient=classi.direction['sgd'])
    # dsngd = dsngd.DSNGD(classi.direction['dsngd'], classi.dual_estimator['dsngd'])
    # dsngd2 = dsngd2.DSNGD2(classi.direction['dsngd2'], classi.dual_estimator['dsngd'])
    # algs = [sgd, dsngd]
    graph_3_dims_3_entropies(sample_length=10000000, many_experiments=100, epochs=1, batch=500)

    # test_kl_and_fast_kl()
