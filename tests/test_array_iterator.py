import unittest

import numpy as np

from src.data.array_iterator import ArrayBatchIterator


class ArrayBatchIteratorTests(unittest.TestCase):
    def test_batches_cover_requested_prefix_without_shuffle(self):
        x = np.arange(10).reshape(5, 2)
        y = np.arange(5)

        batches = list(ArrayBatchIterator(x, y, batch=2, shuffle=False, max_samples=5))

        self.assertEqual(len(batches), 3)
        np.testing.assert_array_equal(np.vstack([batch_x for batch_x, _ in batches]), x)
        np.testing.assert_array_equal(np.concatenate([batch_y for _, batch_y in batches]), y)

    def test_same_seed_short_run_is_prefix_of_long_run(self):
        x = np.arange(20).reshape(10, 2)
        y = np.arange(10)

        short_batches = list(ArrayBatchIterator(x, y, batch=2, random_seed=7, max_samples=4))
        long_batches = list(ArrayBatchIterator(x, y, batch=2, random_seed=7, max_samples=8))
        short_x = np.vstack([batch_x for batch_x, _ in short_batches])
        long_x = np.vstack([batch_x for batch_x, _ in long_batches])

        np.testing.assert_array_equal(short_x, long_x[: len(short_x)])

    def test_rejects_invalid_batch_settings(self):
        x = np.arange(10).reshape(5, 2)
        y = np.arange(5)

        with self.assertRaisesRegex(ValueError, "batch"):
            ArrayBatchIterator(x, y, batch=0)


if __name__ == "__main__":
    unittest.main()
