import gzip
import os
import struct
import tempfile
import unittest

import numpy as np

from src.data.array_iterator import ArraySampleIterator
from src.data.datasets import read_idx


class ArraySampleIteratorTests(unittest.TestCase):
    def test_epochs_cover_the_data_in_batches(self):
        x, y = np.arange(20).reshape(10, 2), np.arange(10)
        sample = ArraySampleIterator(x, y, epochs=2, batch=4, random_seed=0)
        batches = list(sample)

        self.assertEqual(len(sample), len(batches))
        self.assertEqual([len(b[1]) for b in batches], [4, 4, 2, 4, 4, 2])
        for epoch in (batches[:3], batches[3:]):
            np.testing.assert_array_equal(np.sort(np.concatenate([b[1] for b in epoch])), y)
        for x_batch, y_batch in batches:
            np.testing.assert_array_equal(x_batch[:, 0], 2 * y_batch)
        np.testing.assert_array_equal(np.concatenate([b[1] for b in list(sample)]),
                                      np.concatenate([b[1] for b in batches]))

    def test_epoch_length_uses_part_of_the_data(self):
        sample = ArraySampleIterator(np.zeros((10, 1)), np.arange(10), batch=3, epoch_length=7)
        self.assertEqual([len(b[1]) for b in sample], [3, 3, 1])


class ReadIdxTests(unittest.TestCase):
    def test_reads_images_and_labels(self):
        images = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            images_path, labels_path = os.path.join(directory, 'images.gz'), os.path.join(directory, 'labels.gz')
            with gzip.open(images_path, 'wb') as file:
                file.write(struct.pack('>IIII', 2051, 2, 3, 4) + images.tobytes())
            with gzip.open(labels_path, 'wb') as file:
                file.write(struct.pack('>II', 2049, 2) + bytes([7, 1]))
            np.testing.assert_array_equal(read_idx(images_path), images)
            np.testing.assert_array_equal(read_idx(labels_path), [7, 1])


if __name__ == '__main__':
    unittest.main()
