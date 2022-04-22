from algorithms.sgd import SGD
import numpy as np


class AdaGrad(SGD):
    def __init__(self, gradient, batch=1):
        super(AdaGrad, self).__init__(gradient)
        self.key = 'adaGrad'
        self.single_learning_rate_parameter = True
        self.run_algorithm = self.adagrad
        self.learning_rate = self.adagrad_learning_rate
        self.batch = batch

    def adagrad(self, data, progress_bar=False):
        sample = data['sample']
        lr = data["lr"]
        current_parameter = data['start'].copy()
        epochs = data['epochs']

        n = len(sample)
        self.set_run(n, epochs)
        parameters = []
        Gt = 0
        sample_indexes = np.arange(n)
        for e in range(epochs):
            it = np.arange(self.iter_per_epoch)
            if progress_bar:
                it = self.tqdm(it)
            for i in it:
                indexes = sample_indexes[i * self.batch:(i + 1) * self.batch]
                # obs = sample[i * self.batch:(i + 1) * self.batch]
                obs = sample[indexes]
                it = e * self.iter_per_epoch + i
                if self.save_current_parameter(it):
                    parameters.append(current_parameter.copy())
                g = self.direction(obs, current_parameter)
                Gt += g*g

                r = self.learning_rate(Gt, lr)
                current_parameter -= r * g
            np.random.shuffle(sample_indexes)
        parameters.append(current_parameter.copy())
        return np.array(parameters)