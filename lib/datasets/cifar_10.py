from __future__ import division

import os
from six.moves import xrange

import numpy as np

from .dataset import Datasets, Dataset
from .download import maybe_download_and_extract

try:
    import cPickle as pickle
except:
    import pickle

URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def _preprocess_images(images):
    images = np.reshape(images, (-1, 3, 32, 32))
    images = np.transpose(images, (0, 2, 3, 1))
    return (1 / 255) * images.astype(np.float32)


def _preprocess_labels(labels, num_labels):
    labels = np.array(labels, np.uint8)
    size = labels.shape[0]
    index_offset = np.arange(size) * num_labels
    labels_one_hot = np.zeros((size, num_labels), np.uint8)
    labels_one_hot.flat[index_offset + labels] = 1
    return labels_one_hot


def _load_batch(data_dir, name):
    path = os.path.join(data_dir, name)
    batch = pickle.load(open(path, 'rb'), encoding='latin1')
    return batch['data'], batch['labels']


class Cifar10(Datasets):
    def __init__(self, data_dir, val_size=5000):
        maybe_download_and_extract(URL, data_dir)
        data_dir = os.path.join(data_dir, 'cifar-10-batches-py')

        images = np.zeros((0, 32 * 32 * 3), dtype=np.float32)
        labels = np.zeros((0), dtype=np.float32)
        for i in xrange(1, 6):
            batch = _load_batch(data_dir, 'data_batch_{}'.format(i))
            images = np.concatenate((images, batch[0]), axis=0)
            labels = np.concatenate((labels, batch[1]), axis=0)

        images = _preprocess_images(images)
        labels = _preprocess_labels(labels, self.num_labels)

        train = Dataset(images[val_size:], labels[val_size:])
        val = Dataset(images[:val_size], labels[:val_size])

        images, labels = _load_batch(data_dir, 'test_batch')
        images = _preprocess_images(images)
        labels = _preprocess_labels(labels, self.num_labels)
        test = Dataset(images, labels)

        super(Cifar10, self).__init__(train, val, test)

    @property
    def labels(self):
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ]
