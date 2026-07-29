import numpy as np

from src.data.batch_iterator import BatchedIterator


class ArrayBatchIterator(BatchedIterator):
    """Mini-batch iterator over in-memory arrays."""

    def __init__(self, x, y, batch=500, epochs=1, shuffle=True, random_seed=1, max_samples=None):
        x = np.asarray(x)
        y = np.asarray(y, dtype=int)
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) == 0:
            raise ValueError("at least one sample is required")

        self.x = x
        self.y = y
        self.shuffle = bool(shuffle)
        self.random_seed = random_seed
        epoch_length = len(x) if max_samples is None else min(int(max_samples), len(x))
        super().__init__(epoch_length=epoch_length, epochs=epochs, batch=batch)

    def _reset(self):
        self.rng = np.random.default_rng(self.random_seed)

    def _batch(self, start, end):
        indices = self.indices[start:end]
        return self.x[indices], self.y[indices]

    def _start_epoch(self):
        if self.shuffle:
            self.indices = self.rng.permutation(len(self.x))[: self.epoch_length]
        else:
            self.indices = np.arange(self.epoch_length)
