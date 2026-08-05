import numpy as np

from src.data.batch_iterator import BatchedIterator
from src.model.joint_mlr import JointMLR


class JointMLRSampleIterator(BatchedIterator):
    def __init__(self, j_mlr: JointMLR, epoch_length, epochs=1, batch=500, random_seed=1, random_seed_epoch=1):
        self.model = j_mlr
        self.random_seed = random_seed
        self.random_seed_epoch = random_seed_epoch
        super().__init__(epoch_length=epoch_length, epochs=epochs, batch=batch)

    def _reset(self):
        if self.model.pyx is None:
            self.model.set_pyx()
        pyx = self.model.pyx.flatten()
        np.random.seed(self.random_seed)
        self.indices = np.random.choice(range(len(pyx)), self.epoch_length, p=pyx)
        self.x = np.array(list(self.model.compute_all_x()))

    def _start_epoch(self):
        if self.current_epoch > 0:
            np.random.seed(self.random_seed_epoch + self.current_epoch)
            np.random.shuffle(self.indices)

    def _batch(self, start, end):
        batch_indices = self.indices[start:end]
        x_indices = batch_indices // self.model.S.many_values
        x = self.x[x_indices]
        y = batch_indices % self.model.S.many_values
        return x, y
