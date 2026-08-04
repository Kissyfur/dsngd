import numpy as np

from src.data.batch_iterator import BatchedIterator
from src.model.naive_bayes_ef import NaiveBayesEF


class NaiveBayesEFSampleIterator(BatchedIterator):
    """Synthetic sampler for Naive Bayes exponential-family models."""

    def __init__(self, model: NaiveBayesEF, epoch_length, epochs=1, batch=500, random_seed=1):
        self.model = model
        self.random_seed = random_seed
        super().__init__(epoch_length=epoch_length, epochs=epochs, batch=batch)

    def _reset(self):
        self.rng = np.random.default_rng(self.random_seed)
        self._class_probabilities = self.model.class_probabilities()
        self._expectation_blocks = [
            family.expectation_from_natural(block.T)
            for family, block in zip(self.model.families, self.model.beta_blocks)
        ]

    def _batch(self, start, end):
        return self._sample_batch(end - start)

    def _sample_batch(self, batch_size):
        y = self.rng.choice(self.model.many_classes, size=batch_size, p=self._class_probabilities)
        x = np.zeros((batch_size, self.model.observation_dim), dtype=float)

        for class_index in range(self.model.many_classes):
            rows = np.nonzero(y == class_index)[0]
            if len(rows) == 0:
                continue
            for feature_index, (family, columns) in enumerate(self.model.family_slices()):
                expectation = self._expectation_blocks[feature_index][class_index]
                draws = np.asarray(family.sample(expectation, self.rng, size=len(rows)), dtype=float)
                x[rows, columns] = draws.reshape(len(rows), family.input_dim)

        return x, y
