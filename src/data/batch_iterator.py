import math


class BatchedIterator:
    """Base class for epoch-based mini-batch iterators."""

    def __init__(self, epoch_length, epochs=1, batch=500):
        self.epoch_length = int(epoch_length)
        self.epochs = int(epochs)
        if self.epoch_length <= 0:
            raise ValueError("epoch_length must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        self.batch = int(batch) if batch < self.epoch_length else self.epoch_length
        if self.batch <= 0:
            raise ValueError("batch must be positive")

    def __iter__(self):
        self.current_epoch = 0
        self.current_epoch_length = 0
        self._reset()
        self._start_epoch()
        return self

    def __next__(self):
        if self.current_epoch_length >= self.epoch_length:
            self.current_epoch += 1
            if self.current_epoch >= self.epochs:
                raise StopIteration
            self.current_epoch_length = 0
            self._start_epoch()

        start = self.current_epoch_length
        end = min(start + self.batch, self.epoch_length)
        self.current_epoch_length = end
        return self._batch(start, end)

    def __len__(self):
        return self.epochs * math.ceil(self.epoch_length / self.batch)

    def _reset(self):
        pass

    def _start_epoch(self):
        pass

    def _batch(self, start, end):
        raise NotImplementedError
