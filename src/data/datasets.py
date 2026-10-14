import gzip
import os
import struct
from urllib.request import urlretrieve

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'saved_data')

COVERTYPE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
MNIST_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
MNIST_FILES = {'train': ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
               'test': ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')}


def download(url, file_name, data_dir=DATA_DIR):
    path = os.path.join(data_dir, file_name)
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        print(f'Downloading {url}')
        urlretrieve(url, path)
    return path


def standardize(x_train, x_test, columns):
    mean, std = x_train[:, columns].mean(axis=0), x_train[:, columns].std(axis=0)
    x_train[:, columns] = (x_train[:, columns] - mean) / std
    x_test[:, columns] = (x_test[:, columns] - mean) / std


def load_covertype(data_dir=DATA_DIR, test_fraction=0.2, seed=0):
    """UCI Covertype: 10 continuous features, wilderness area (4 values) and soil type (40 values), 7 classes.

    Returns x_train, y_train, x_test, y_test. The continuous features are standardized with the train set.
    """
    data = np.loadtxt(download(COVERTYPE_URL, 'covtype.data.gz', data_dir), delimiter=',', dtype=np.int64)
    x = np.column_stack([data[:, :10], np.argmax(data[:, 10:14], axis=1), np.argmax(data[:, 14:54], axis=1)])
    x = x.astype(float)
    y = data[:, 54] - 1
    order = np.random.default_rng(seed).permutation(len(y))
    n_test = int(test_fraction * len(y))
    x_train, y_train, x_test, y_test = x[order[n_test:]], y[order[n_test:]], x[order[:n_test]], y[order[:n_test]]
    standardize(x_train, x_test, slice(0, 10))
    return x_train, y_train, x_test, y_test


def read_idx(path):
    with gzip.open(path, 'rb') as file:
        magic, count = struct.unpack('>II', file.read(8))
        shape = [count] + [struct.unpack('>I', file.read(4))[0] for _ in range((magic & 0xff) - 1)]
        return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)


def load_mnist(binary=False, data_dir=DATA_DIR, min_active=0.01):
    """MNIST with one feature per pixel; pixels active in less than min_active of the train images are removed.

    Pixels are standardized with the train set, or binarized (value >= 128) when binary is True.
    """
    splits = []
    for images, labels in (MNIST_FILES['train'], MNIST_FILES['test']):
        x = read_idx(download(MNIST_URL + images, images, data_dir)).reshape(-1, 28 * 28).astype(float)
        splits.append((x, read_idx(download(MNIST_URL + labels, labels, data_dir)).astype(int)))
    (x_train, y_train), (x_test, y_test) = splits
    pixels = np.mean(x_train > 0, axis=0) >= min_active
    x_train, x_test = x_train[:, pixels], x_test[:, pixels]
    if binary:
        return (x_train >= 128).astype(float), y_train, (x_test >= 128).astype(float), y_test
    standardize(x_train, x_test, slice(None))
    return x_train, y_train, x_test, y_test
