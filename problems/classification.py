import numpy as np
import itertools

from algorithms import map as mAp
from scipy.special import logsumexp
from problems import utils as u

# import sample_manager as sm
# from datasets import dataset_loader
# np.set_printoptions(threshold=np.nan)    


class CanonicalParametrization(object):
    def __init__(self, y_values, x_values):
        self.y_values = y_values
        self.xd_values = np.array(x_values["discrete"])
        self.xg_value = x_values["gaussian"]
        self.block_size = 500000
        self.block_size2 = 10000


        self.many_xd = len(self.xd_values)
        self.many_xg = int(self.xg_value)
        # self.k = self.many_xd + self.many_xc

        self.xg_dimension = 2 * self.many_xg
        self.xd_dimension = int(np.sum(self.xd_values)) - self.many_xd
        self.x_dimension = self.xd_dimension + self.xg_dimension + 1
        self.y_dimension = self.y_values
        self.many_x = np.prod(self.xd_values)

        self.dimension = self.y_dimension * self.x_dimension

        self.zero_indexes = self.find_zero_indexes()
        print('dimension is: ', self.dimension)
        self.all_x_iter = self.compute_all_x()
        self.all_x = np.array(list(self.all_x_iter()))
        # self.all_tx_big = self.big_t_x(self.all_x)
        # self.all_tx = self.canonical_tx(self.all_tx_big)
        self.pyx = None

    def canonical_tx(self, tx):
        return np.delete(tx, self.zero_indexes, axis=1)

    def canonical_sample(self, sample):
        return np.delete(sample, self.zero_indexes + 1, axis=1)

    @staticmethod
    def identity_it(arr, dimension):
        arr = arr.copy().astype(int)
        id_y = np.identity(dimension, dtype=int)
        e_arr = id_y[arr]
        return e_arr

    def s_y(self, y):
        sy = self.identity_it(y, self.y_values)[:, 1:]
        return sy

    def t_x(self, x):
        many_x = len(x)
        tx = np.ones((many_x, self.x_dimension))
        start = 1
        for i in range(self.many_xd):
            end = start + self.xd_values[i]-1
            tx[:, start:end] = self.identity_it(x[:, i], self.xd_values[i])[:, 1:]
            start = end
        # tx[:, start:] = x[:, self.many_xd:]
        return tx

    def big_t_x(self, x):
        many_x = len(x)
        tx = np.ones((many_x, self.x_dimension+self.many_xd))
        start = 1
        for i in range(self.many_xd):
            end = start + self.xd_values[i]
            tx[:, start:end] = self.identity_it(x[:, i], self.xd_values[i])
            start = end
        tx[:, start:] = x[:, self.many_xd:]
        return tx

    def tx_dsngd(self, tx):
        tx_m = tx.copy()
        z = tx[:, self.zero_indexes]
        for i in range(self.many_xd):
            xi_is_0 = np.array(np.nonzero(z[:, i]))
            tx_m[xi_is_0, self.zero_indexes[i]:self.zero_indexes[i]+self.xd_values[i]] = np.ones(self.xd_values[i])
        tx_m = self.canonical_tx(tx_m)
        return tx_m

    def compute_all_x(self):
        ranged_xd_values = [range(i) for i in self.xd_values]

        def new_iterator():
            return itertools.product(*ranged_xd_values)
        return new_iterator

    def find_zero_indexes(self):
        if self.many_xd == 0:
            return []
        z = [1]
        for i in range(self.many_xd - 1):
            z.append(z[i] + self.xd_values[i])
        return np.array(z)

    def compute_log_numerators(self, tx, params):
        params_copy = params.copy()
        params_copy[:, 0, 0] = 0
        # params_copy = np.reshape(params_copy, (n, self.y_values, self.x_dimension))
        log_numerators = np.dot(params_copy, tx.T)
        return log_numerators

    def all_log_p(self, tx, betas):
        log_numerators = self.compute_log_numerators(tx, betas)
        all_log_p = self.compute_all_log_p(log_numerators)
        return all_log_p

    def compute_all_log_p(self, log_numerators):
        log_denominators = logsumexp(log_numerators, axis=1)
        log_denominators = np.where(log_denominators == -np.inf, 0., log_denominators)
        for i in range(self.y_values):
            log_numerators[:, i, :] -= log_denominators
        return log_numerators

    # compute logarithms of P(Y|X) of a sample for a group of parameters 
    def log_p(self, sample, params):
        y = sample[:, 0]
        tx = sample[:, 1:]
        all_log_p = self.all_log_p(tx, params)
        n = len(y)
        y = y.copy().astype(int)
        log_p = all_log_p[:, y, range(n)]
        return log_p

    def set_pyx(self, parameter):
        log_num = []
        q = self.many_x // self.block_size + 1
        r = self.many_x % self.block_size
        all_x = self.all_x_iter()
        for i in range(q):
            if i == i - 1:
                last = r
            else:
                last = self.block_size
            x_block = np.array(list(itertools.islice(all_x, 0, last)))
            tx_block = self.t_x(x_block)
            log_numerators = self.compute_log_numerators(tx_block, parameter)
            log_num.append(log_numerators[0])
        log_num = np.hstack(log_num)
        log_denominator = logsumexp(log_num)
        pyx = np.exp(log_num - log_denominator)
        self.pyx = pyx
        return

    def compute_px_given_parameters(self, params):
        log_numerators = self.compute_log_numerators(self.all_tx, params)
        log_denominator = logsumexp(log_numerators)
        pyx = np.exp(log_numerators - log_denominator)
        px = np.sum(pyx, axis=1)
        return px

    def kl_divergence(self, params):
        q = self.many_x // self.block_size2 + 1
        r = self.many_x % self.block_size2
        all_x = self.all_x_iter()
        kl = 0
        for i in range(q):
            if i == q - 1:
                last = r
            else:
                last = self.block_size2
            if last == 0:
                break
            x_block = np.array(list(itertools.islice(all_x, 0, last)))
            tx_block = self.t_x(x_block)
            log_q = self.all_log_p(tx_block, params)
            qs = np.exp(log_q)
            p_block = self.pyx[:, i * self.block_size2: i * self.block_size2 + last]
            kl += u.kl_divergence(p_block, qs)
        return kl
    
# class DualParametrization: #implemented only for discrete case
#     def __init__(self, y_values, x_values):
#         self.y_values = y_values
#         self.xd_values = np.array(x_values["discrete"])
#         self.xg_value = x_values["gaussian"]
#
#         self.many_xd = len(self.xd_values)
#         self.many_xg = int(self.xg_value)
#         # self.k = self.many_xd + self.many_xc
#
#         self.xg_dimension = self.many_xg
#         # self.xd_dimension = int(np.sum(self.xd_values))
#         self.xd_dimension = int(np.sum(np.subtract(self.xd_values, 1)))
#         self.x_dimension = self.xd_dimension + self.xg_dimension + 1
#         self.y_dimension = self.y_values
#
#         self.dimension = self.y_dimension * self.x_dimension


class ClassificationProblem:

    def __init__(self, var, dataset=False):
    
        self.y_values = var["Y"]
        self.x_values = var["X"]
        self.xd_values = np.array(self.x_values["discrete"])
        self.xc_value = self.x_values["gaussian"]
        print('classes: ', self.y_values, '     discrete variables: ', self.xd_values,
              '        continuous variables: ', self.xc_value)

        self.cp = CanonicalParametrization(self.y_values, self.x_values)
        self.sample_block = 100000

        self.starting_parameter = self.max_entropy_parameter()
        self.dual_starting_parameter = self.max_entropy_dual_parameter()

        self.cost_function = self.minus_conditional_log_likelihood_to_sample
        self.training_lr_cost_function = self.minus_cumulative_conditional_log_likelihood
        self.direction = {'sgd': self.canonical_parametrization_gradient,
                          'dsngd': self.dsngd_direction,
                          'dsngd2': self.dsngd_direction2,
                          'csngd': self.csngd_direction
                          }
        self.start = {'sgd': self.starting_parameter,
                      'dsngd': self.starting_parameter,
                      'csngd': self.starting_parameter}
        self.dual_start = {'dsngd': self.dual_starting_parameter}
        self.dual_estimator = {'dsngd': mAp.MAP(self)}

    def set_problem(self, sigma, seed):
        dim = (1, self.cp.y_dimension, self.cp.x_dimension)
        np.random.seed(seed)
        real_parameter = np.random.normal(0., sigma, size=dim)
        return real_parameter

    # def generate_random_sample(self, n):
    #     np.random.seed(n)
    #     py = np.random.random(self.cp.y_dimension)
    #     py /= np.sum(py)
    #     y = np.random.choice(range(self.cp.y_dimension), n, p=py)
    #     # px = np.random.random(classi.cp.y_dimension)
    #     xT = []
    #     for i in range(self.cp.many_xd):
    #         p = np.random.random(self.cp.xd_values[i])
    #         p /= np.sum(p)
    #         xT.append(np.random.choice(range(self.cp.xd_values[i]), n, p=p))
    #     x = np.array(xT).T
    #     big_tx = self.cp.big_t_x(x)
    #     sample_big = np.vstack((y, big_tx.T)).T
    #     return sample_big

    def generate_sample(self, sample_length, epochs=1, batch=1, seed=0, shuffle_seed=0):
        b = int(sample_length // self.sample_block) + 1
        r = int(sample_length % self.sample_block)
        pyx = self.cp.pyx
        if np.any(pyx == None):
            print("no parameter selected for sample")
            exit()
        pyx = pyx.flatten()

        def new_iterator():
            for e in range(epochs):
                for block in range(b):
                    np.random.seed(block + seed)
                    if block == b-1:
                        many_obs = r
                    else:
                        many_obs = self.sample_block
                    if many_obs == 0:
                        break
                    indices = np.random.choice(range(len(pyx)), many_obs, p=pyx)
                    np.random.seed(e + shuffle_seed)
                    np.random.shuffle(indices)
                    x_indices = indices % self.cp.many_x
                    y = indices // self.cp.many_x
                    x = self.cp.all_x[x_indices]
                    tx = self.cp.t_x(x)
                    sample = np.vstack((y, tx.T)).T
                    for j in range(0, many_obs, batch):
                        yield sample[j:j + batch]
        return new_iterator

    def generate_sample_big(self, sample_length, epochs=1, batch=1, seed=0, shuffle_seed=0):
        b = int(sample_length // self.sample_block) + 1
        r = int(sample_length % self.sample_block)
        pyx = self.cp.pyx
        if np.any(pyx == None):
            print("no parameter selected for sample")
            exit()
        pyx = pyx.flatten()

        def new_iterator():
            for e in range(epochs):
                for block in range(b):
                    np.random.seed(block + seed)
                    if block == b-1:
                        many_obs = r
                    else:
                        many_obs = self.sample_block
                    if many_obs == 0:
                        break
                    indices = np.random.choice(range(len(pyx)), many_obs, p=pyx)
                    np.random.seed(e + shuffle_seed)
                    np.random.shuffle(indices)
                    x_indices = indices % self.cp.many_x
                    y_indices = indices // self.cp.many_x
                    y = np.arange(self.cp.y_values)[y_indices]
                    x = self.cp.all_x[x_indices]
                    tx = self.cp.big_t_x(x)
                    sample = np.vstack((y, tx.T)).T
                    for j in range(0, many_obs, batch):
                        yield sample[j:j + batch]
        return new_iterator

    def generate_sample_x_and_ygx(self, n, real_parameter, seed=0):
        np.random.seed(seed)

        def compute_px_given_parameters(params):
            all_x = self.cp.compute_all_x()
            all_tx = self.cp.t_x(all_x)
            log_numerators = self.cp.compute_log_numerators(all_tx, params)
            log_denominator = logsumexp(log_numerators)
            pyx = np.exp(log_numerators - log_denominator)
            px = np.sum(pyx, axis=1)
            return px

        px = compute_px_given_parameters(real_parameter)[0]
        all_x = self.cp.compute_all_x()
        x_indices = np.random.choice(range(len(all_x)), n, p=px)
        x = all_x[x_indices]
        tx = self.cp.t_x(x)

        pygx = np.exp(self.cp.all_log_p(tx, real_parameter)[0])
        y = []
        for i in range(n):
            y.append(np.random.choice(self.y_values, p=pygx[:, i]))
        y = np.array(y)
        big_tx = self.cp.big_t_x(x)
        sample_big = np.vstack((y, big_tx.T)).T
        print("Sample created")
        return sample_big

    ###############################################################################################################

    # def minus_conditional_log_likelihood(self, sample, parameters):
    #     cll = u.cll(self.cp.log_p, sample, parameters)
    #     return -cll

    def minus_conditional_log_likelihood_to_sample(self, sample):

        def minus_cll(parameters):
            return -u.cll(self.cp.log_p, sample, parameters)
        return minus_cll

    def minus_cumulative_conditional_log_likelihood(self, sample, parameters):
        sam = self.cp.canonical_sample(sample.copy())
        ncll = u.ncll(self.cp.log_p, sam, parameters)
        return -ncll

    def kl_to_real_parameter(self, real_parameter):
        space = []
        for num in self.xd_values:
            space.append(np.arange(num))
        gen_tx = u.itertools.product(*space)
        X = np.array(list(gen_tx))
        TX = self.cp.t_x(X)
        log_p_real = self.cp.compute_log_numerators(TX, real_parameter)
        log_p_real = log_p_real - logsumexp(log_p_real)
        p_real = np.exp(log_p_real)

        # print("Testing the pyx: ", np.sum(np.abs(p_real-self.cp.pyx)))
        def kl_divergence(params):
            log_q = self.cp.all_log_p(TX, params)
            qs = np.exp(log_q)
            return u.kl_divergence(p_real[0], qs)

        return kl_divergence

###############################################################################################################

    def q_minus_e(self, sample, param):
        y = sample[:, 0]
        t_x = sample[:, 1:]
        canonical_y = self.cp.identity_it(y, self.y_values)
        param = np.reshape(param, (1,) + param.shape)
        log_q_Y = self.cp.all_log_p(t_x, param)[0]
        q_Y_minus_e = np.exp(log_q_Y)
        q_Y_minus_e -= canonical_y.T
        return q_Y_minus_e

    def canonical_parametrization_gradient(self, sample, parameter):
        tx = sample[:, 1:]
        q_Y_minues_e = self.q_minus_e(sample, parameter)
        g = np.dot(q_Y_minues_e, tx)
        # g[0, 0] = 0
        return g

    def dsngd_direction_old(self, sample, sample_cp, param, dual_param):
        y = sample[:, 0]
        t_y = self.cp.identity_it(y, self.y_values)
        param = np.reshape(param, ((1,)+ param.shape))
        tx = sample[:, 1:]
        tx_dsngd = self.cp.tx_dsngd(tx)

        tx_cp = np.delete(tx, self.cp.zero_indexes, axis=1)

        log_p = self.cp.all_log_p(tx_cp, param)[0]
        p = np.exp(log_p)
        p = (p.T-t_y).T

        g = [np.dot(p, tx_dsngd)]
        ng = 0
        i=0

        dual_param_normalized = dual_param.copy()
        dual_param_normalized = dual_param_normalized/np.sum(dual_param[:, 0])
        for gi in g:
            hi = np.dot(np.ones((self.y_values,1)), tx_cp[i:i+1])

            hi[:, 0] *= (1 - self.cp.many_xd)
            hi /= dual_param_normalized

            dual_param_normalized_z = np.ones((self.cp.y_dimension, self.cp.many_xd))
            start = 1

            for k in range(self.cp.many_xd):
                end = start + self.xd_values[k]-1
                dual_param_normalized_z[:,k] = dual_param_normalized[:, 0]
                dual_param_nz = dual_param_normalized[:,start:end ]
                dual_param_normalized_z[:,k] -= np.sum(dual_param_nz, axis=1)
                start = end
            inv_z = np.divide(1., dual_param_normalized_z)
            xk_zeros = np.nonzero(tx[i][self.cp.zero_indexes])[0]
            hi[:,0] += np.sum(inv_z[:, xk_zeros], axis=1)
            tx0 = tx[i, self.cp.zero_indexes]
            start = 1
            for k in range(self.cp.many_xd):
                end = start + self.xd_values[k]-1
                if tx0[k]!= 0:
                    xk_is_0 = np.nonzero(tx0[k])[0]
                    fill = (np.ones((self.xd_values[k]-1, self.cp.y_dimension)) * inv_z[:, k]).T
                    hi[:, start:end] = -fill
                start = end
            ng += gi * hi
            i += 1
        ng[:, 0] -= ng[0,0]
        return ng

    def dsngd_direction2(self, sample, sample_cp, parameter, dual_parameter):
        # tx = sample[:,1:]
        tx_cp = sample_cp[:,1:]

        q_Y_minus_e = self.q_minus_e(sample_cp, parameter)
        dual_parameter_normalized = dual_parameter/np.sum(dual_parameter[:,0])
        g = (np.dot(q_Y_minus_e, tx_cp)/ dual_parameter_normalized)
        return g

    def dsngd_direction(self, sample, sample_cp, parameter, dual_parameter):
        # g_old = self.dsngd_direction_old(sample, parameter, dual_parameter)
        tx = sample[:, 1:]
        tx_cp = sample_cp[:, 1:]

        q_Y_minus_e = self.q_minus_e(sample_cp, parameter)
        dual_parameter_normalized = dual_parameter/np.sum(dual_parameter[:, 0])
        g = (np.dot(q_Y_minus_e, tx_cp) / dual_parameter_normalized)
        g[:, 0] *= 1 - self.cp.many_xd

        # q_Y_minus_e_zeros =  q_Y_minus_e.copy()
        tx0 = tx[:, self.cp.zero_indexes]
        start = 1

        h_y = np.ones((len(sample), self.cp.y_dimension)) - self.cp.many_xd
        h_y /= dual_parameter_normalized[:, 0]
        for i in range(self.cp.many_xd):

            end = start + self.xd_values[i]-1
            xi_is_0 = np.nonzero(tx0[:, i])[0]
            txi_zero = np.ones((self.y_values, self.xd_values[i]-1))*(-1)
            dual_param_i_zero = dual_parameter_normalized[:, 0]-np.sum(dual_parameter_normalized[:, start:end], axis=1)
            fill = (1./dual_param_i_zero) * np.sum(q_Y_minus_e[:, xi_is_0], axis=1)

            h_y[xi_is_0] += (1./dual_param_i_zero)
            g[:, start:end] += (txi_zero.T*fill).T
            start = end
        g0 = np.sum(q_Y_minus_e * h_y.T, axis=1)
        g0 -= g0[0]
        g[:, 0] = g0
        # g2 = 0
        # if len(sample)!= 1:
        #     for i in range(len(sample)):
        #         g2 += self.dsngd_direction(sample[i:i+1], sample_cp[i:i+1], parameter, dual_parameter)
        # if np.sum(np.abs(g-g2))>1e8:
        #     print('batch not well computed')
        # if np.sum(np.abs(g-g_old))>1e8:
        #     print('Ahhhh', g-g_old)
        #     exit()
        return g

    def csngd_direction(self, sample, parameter, dual_parameter):
        return

################################################################################################
#
#   Train learning rate
#
#################################################################################################  

    def max_entropy_parameter(self):
        return np.zeros((self.cp.y_dimension, self.cp.x_dimension))

    def max_entropy_dual_parameter(self):
        psi = self.cp.dimension 
        dual_p = np.ones((self.cp.y_dimension, self.cp.x_dimension)) * psi / self.y_values
        start = 1
        for i in range(self.cp.many_xd):
            end = start + (self.xd_values[i])-1
            dual_p[:, start:end] = psi / (self.xd_values[i] * self.y_values)
            start = end
        # dual_p = np.random.random((self.cp.y_dimension,self.dp.x_dimension))
        return dual_p


################################################################################################################

    def get_file_name(self):
        x_name = str(self.xd_values).replace(' ', '') + str(np.array([self.xc_value]))
        return str(self.y_values) + x_name  # + str(int(fm))

    def get_graph_title(self, e, n):
        x_name = str(self.xd_values).replace(' ', '') + str(np.array([self.xc_value]))
        return 'n:' + str(n) + ', y val:' + str(self.y_values) + ', X val:' + x_name + ' ep:'+str(e)


