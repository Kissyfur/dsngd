import numpy as np


class ArraySampleIterator:
    """Batches of a data set (x, y), reshuffled each epoch. With epoch_length, each epoch uses only the first
    epoch_length observations of the shuffled data."""

    def __init__(self, x, y, epochs=1, batch=500, random_seed=1, epoch_length=None):
        self.x = x
        self.y = y
        self.epoch_length = len(y) if epoch_length is None else min(epoch_length, len(y))
        self.epochs = epochs
        self.batch = batch if batch < self.epoch_length else self.epoch_length
        self.random_seed = random_seed

    def __iter__(self):
        self.rng = np.random.default_rng(self.random_seed)
        self.indices = self.rng.permutation(len(self.y))[:self.epoch_length]
        self.current_epoch = 0
        self.current_epoch_length = 0
        return self

    def __next__(self):
        if self.current_epoch_length >= self.epoch_length:
            self.current_epoch_length = 0
            self.current_epoch += 1
            self.indices = self.rng.permutation(len(self.y))[:self.epoch_length]
        if self.current_epoch >= self.epochs:
            raise StopIteration
        batch_indices = self.indices[self.current_epoch_length: self.current_epoch_length + self.batch]
        self.current_epoch_length += self.batch
        return self.x[batch_indices], self.y[batch_indices]

    def __len__(self):
        return self.epochs * int(np.ceil(self.epoch_length / self.batch))
