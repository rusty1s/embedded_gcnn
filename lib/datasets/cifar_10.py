from __future__ import division

import os
import sys
from six.moves import xrange

import numpy as np

from .dataset import Datasets, Dataset
from .download import maybe_download_and_extract

try:
    import cPickle as pickle
except:
    import pickle

URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


class Cifar10(Datasets):
    def __init__(self, data_dir, val_size=5000):
        maybe_download_and_extract(URL, data_dir)
        self.data_dir = os.path.join(data_dir, 'cifar-10-batches-py')

        images = np.zeros(
            (0, self.width * self.height * self.num_channels),
            dtype=np.float32)
        labels = np.zeros((0), dtype=np.uint8)

        for i in xrange(1, 6):
            batch = self._load_batch('data_batch_{}'.format(i))
            images = np.concatenate((images, batch[0]), axis=0)
            labels = np.concatenate((labels, batch[1]), axis=0)

        images = self._preprocess_images(images)
        labels = self._preprocess_labels(labels)

        train = Dataset(images, labels)

        images, labels = self._load_batch('test_batch')
        images = self._preprocess_images(images)
        labels = self._preprocess_labels(labels)
        val = Dataset(images, labels)
        test = Dataset(images, labels)

        super(Cifar10, self).__init__(train, val, test)

    @property
    def classes(self):
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ]

    @property
    def width(self):
        return 32

    @property
    def height(self):
        return 32

    @property
    def num_channels(self):
        return 3

    def _preprocess_images(self, images):
        images = np.reshape(images, (-1, self.num_channels, self.width,
                                     self.height))
        images = np.transpose(images, (0, 2, 3, 1))
        return (1 / 255) * images.astype(np.float32)

    def _preprocess_labels(self, labels):
        # Convert labels to one hot.
        labels = np.array(labels, np.uint8)
        size = labels.shape[0]
        index_offset = np.arange(size) * self.num_classes
        labels_one_hot = np.zeros((size, self.num_classes), np.uint8)
        labels_one_hot.flat[index_offset + labels] = 1
        return labels_one_hot

    def _load_batch(self, name):
        with open(os.path.join(self.data_dir, name), 'rb') as f:
            if sys.version_info >= (3, 0):
                batch = pickle.load(f, encoding='latin1')
            else:  # pragma: no cover
                batch = pickle.load(f)
        return batch['data'], batch['labels']
