import numpy as np

from tqdm import tqdm
from src.model.joint_mlr import JointMLR


class JointMLRSampleIterator:
    def __init__(self, j_mlr: JointMLR, epoch_length, epochs=1, batch=500, random_seed=1, random_seed_epoch=1):
        self.model = j_mlr
        self.epoch_length = epoch_length
        self.epochs = epochs
        self.batch = batch if batch < epoch_length else epoch_length
        self.total_length = self.epochs * self.epoch_length
        self.random_seed = random_seed
        self.random_seed_epoch = random_seed_epoch

    def __iter__(self):
        if self.model.pyx is None:
            self.model.set_pyx()
        pyx = self.model.pyx.flatten()
        np.random.seed(self.random_seed)
        self.indices = np.random.choice(range(len(pyx)), self.epoch_length, p=pyx)
        self.current_length = 0
        self.current_epoch = 0
        self.current_epoch_length = 0
        self.x = np.array(list(self.model.compute_all_x()))
        return self

    def __next__(self):
        if self.current_epoch_length >= self.epoch_length:
            self.current_epoch_length = 0
            self.current_epoch += 1
            np.random.seed(self.random_seed_epoch + self.current_epoch)
            np.random.shuffle(self.indices)
        if self.current_epoch >= self.epochs:
            raise StopIteration
        batch_indices = self.indices[self.current_epoch_length: self.current_epoch_length + self.batch]
        x_indices = batch_indices // self.model.S.many_values
        x = self.x[x_indices]
        y = batch_indices % self.model.S.many_values
        self.current_epoch_length += self.batch
        return x, y

    def __len__(self):
        return self.total_length // self.batch


if __name__ == "__main__":
    j_mlr = JointMLR(3, [4, 3, 2])
    j_mlr_sampler = JointMLRSampleIterator(j_mlr, 10000000, 2, 500)
    x = 0
    for obs in tqdm(j_mlr_sampler):
        x += 1

    for obs in tqdm(j_mlr_sampler):
        x += 1
    print("finished")
