import math

import numpy as np


class ArrayBatchIterator:
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
        self.epochs = int(epochs)
        self.shuffle = bool(shuffle)
        self.random_seed = random_seed
        self.epoch_length = len(x) if max_samples is None else min(int(max_samples), len(x))
        if self.epoch_length <= 0:
            raise ValueError("max_samples must be positive")
        self.batch = int(batch) if batch < self.epoch_length else self.epoch_length

    def __iter__(self):
        self.rng = np.random.default_rng(self.random_seed)
        self.current_epoch = 0
        self.current_epoch_length = 0
        self._start_epoch()
        return self

    def __next__(self):
        if self.current_epoch_length >= self.epoch_length:
            self.current_epoch += 1
            if self.current_epoch >= self.epochs:
                raise StopIteration
            self.current_epoch_length = 0
            self._start_epoch()

        end = min(self.current_epoch_length + self.batch, self.epoch_length)
        indices = self.indices[self.current_epoch_length:end]
        self.current_epoch_length = end
        return self.x[indices], self.y[indices]

    def __len__(self):
        return self.epochs * math.ceil(self.epoch_length / self.batch)

    def _start_epoch(self):
        if self.shuffle:
            self.indices = self.rng.permutation(len(self.x))[: self.epoch_length]
        else:
            self.indices = np.arange(self.epoch_length)
