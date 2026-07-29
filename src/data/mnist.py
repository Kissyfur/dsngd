import gzip
import struct
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np


MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES.values():
        path = data_dir / filename
        if path.exists():
            continue
        url = f"{MNIST_URL}/{filename}"
        print(f"Downloading {url} to {path}", flush=True)
        urlretrieve(url, path)


def load_mnist(data_dir, download=True):
    data_dir = Path(data_dir)
    if download:
        download_mnist(data_dir)

    x_train = read_idx_images(data_dir / MNIST_FILES["train_images"])
    y_train = read_idx_labels(data_dir / MNIST_FILES["train_labels"])
    x_test = read_idx_images(data_dir / MNIST_FILES["test_images"])
    y_test = read_idx_labels(data_dir / MNIST_FILES["test_labels"])
    return x_train, y_train, x_test, y_test


def read_idx_images(path):
    with gzip.open(path, "rb") as file:
        magic, count, rows, columns = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data.reshape(count, rows * columns)


def read_idx_labels(path):
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        data = np.frombuffer(file.read(), dtype=np.uint8)
    return data[:count].astype(int)
