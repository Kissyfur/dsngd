import math

import numpy as np

from src.model.naive_bayes_ef import NaiveBayesEF


class NaiveBayesEFSampleIterator:
    """Synthetic sampler for Naive Bayes exponential-family models."""

    def __init__(self, model: NaiveBayesEF, epoch_length, epochs=1, batch=500, random_seed=1):
        self.model = model
        self.epoch_length = int(epoch_length)
        self.epochs = int(epochs)
        self.batch = int(batch) if batch < epoch_length else int(epoch_length)
        self.random_seed = random_seed

    def __iter__(self):
        self.rng = np.random.default_rng(self.random_seed)
        self.current_epoch = 0
        self.current_epoch_length = 0
        self._class_probabilities = self.model.class_probabilities()
        self._expectation_blocks = [
            family.expectation_from_natural(block.T)
            for family, block in zip(self.model.families, self.model.beta_blocks)
        ]
        return self

    def __next__(self):
        if self.current_epoch_length >= self.epoch_length:
            self.current_epoch += 1
            self.current_epoch_length = 0
        if self.current_epoch >= self.epochs:
            raise StopIteration

        batch_size = min(self.batch, self.epoch_length - self.current_epoch_length)
        self.current_epoch_length += batch_size
        return self._sample_batch(batch_size)

    def __len__(self):
        return self.epochs * math.ceil(self.epoch_length / self.batch)

    def _sample_batch(self, batch_size):
        y = self.rng.choice(self.model.many_classes, size=batch_size, p=self._class_probabilities)
        x = np.zeros((batch_size, len(self.model.families)), dtype=float)

        for class_index in range(self.model.many_classes):
            rows = np.nonzero(y == class_index)[0]
            if len(rows) == 0:
                continue
            for feature_index, family in enumerate(self.model.families):
                expectation = self._expectation_blocks[feature_index][class_index]
                x[rows, feature_index] = family.sample(expectation, self.rng, size=len(rows))

        return x, y
