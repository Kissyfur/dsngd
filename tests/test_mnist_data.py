import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.mnist import read_idx_images, read_idx_labels


class MNISTDataTests(unittest.TestCase):
    def test_reads_idx_images_and_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            image_path = temp_dir / "images.gz"
            label_path = temp_dir / "labels.gz"

            with gzip.open(image_path, "wb") as file:
                file.write(struct.pack(">IIII", 2051, 2, 2, 2))
                file.write(np.arange(8, dtype=np.uint8).tobytes())
            with gzip.open(label_path, "wb") as file:
                file.write(struct.pack(">II", 2049, 2))
                file.write(np.array([3, 7], dtype=np.uint8).tobytes())

            images = read_idx_images(image_path)
            labels = read_idx_labels(label_path)

        np.testing.assert_array_equal(images, np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint8))
        np.testing.assert_array_equal(labels, np.array([3, 7]))


if __name__ == "__main__":
    unittest.main()
