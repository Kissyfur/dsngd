from src.algorithms.adaGrad import AdaGrad
import numpy as np


class Adadelta(AdaGrad):
    def __init__(self, gradient, batch=1):
        super(Adadelta, self).__init__(gradient)
        self.key = 'adadelta'
        self.single_learning_rate_parameter = True
        self.run_algorithm = self.adadelta
        self.learning_rate = self.adadelta_learning_rate
        self.batch = batch

    @staticmethod
    def adadelta_learning_rate(E_gti, E_updates):
        fudge_factor = 1e-8
        RMS_gti = np.sqrt(fudge_factor + E_gti)
        RMS_updates = np.sqrt(fudge_factor + E_updates)
        return RMS_updates / RMS_gti


    def adadelta(self, data, epochs, alpha=0.9, progress_bar=False):
        sample = data["sample"]
        current_parameter = data["start"].copy()

        n = len(sample)
        self.set_run(n, epochs)
        parameters = []
        Gt = 0
        E_increment = 0
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
                Gt = Gt * alpha + (1 - alpha) * (g * g)

                r = self.learning_rate(Gt, E_increment)
                increment = - r * g
                current_parameter += increment
                E_increment = E_increment * alpha + (1 - alpha) * (increment * increment)
            np.random.shuffle(sample_indexes)
        parameters.append(current_parameter.copy())
        return np.array(parameters)