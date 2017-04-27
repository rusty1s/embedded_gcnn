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

LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]


def _preprocess(images):
    images = np.reshape(images, (-1, 3, 32, 32))
    images = np.transpose(images, (0, 2, 3, 1))
    return (1 / 255) * images.astype(np.float32)


def _load_batch(data_dir, name):
    batch = pickle.load(
        open(os.path.join(data_dir, name), 'rb'), encoding='latin1')

    labels = np.array(batch['labels'], np.uint8)
    index_offset = np.arange(labels.shape[0]) * 10
    labels_one_hot = np.zeros((labels.shape[0], 10), np.uint8)
    labels_one_hot.flat[index_offset + labels] = 1

    return batch['data'], labels_one_hot


class Cifar10(Datasets):
    def __init__(self, data_dir, validation_size=5000):
        maybe_download_and_extract(URL, data_dir)
        data_dir = os.path.join(data_dir, 'cifar-10-batches-py')

        # Load the train data from disk.
        train_images = []
        train_labels = []
        for i in xrange(1, 6):
            images, labels = _load_batch(data_dir, 'data_batch_{}'.format(i))
            train_images.append(images)
            train_labels.append(labels)

        train_images = np.stack(train_images, axis=0)
        train_images = np.reshape(train_images, (-1, 32 * 32 * 3))
        train_images = _preprocess(train_images)
        train_labels = np.stack(train_labels, axis=0)
        train_labels = np.reshape(train_labels, (-1, self.num_labels))

        train_images = train_images[validation_size:]
        train_labels = train_labels[validation_size:]
        train = Dataset(train_images, train_labels)

        # Generate a slice of validation data from training data.
        validation_images = train_images[:validation_size]
        validation_labels = train_labels[:validation_size]
        validation = Dataset(validation_images, validation_labels)

        # Load the test data from disk.
        test_images, test_labels = _load_batch(data_dir, 'test_batch')
        test_images = _preprocess(test_images)
        test = Dataset(test_images, test_labels)

        super(Cifar10, self).__init__(train, validation, test)

    def label_name(self, label):
        return [LABELS[i] for i in np.where(label == 1)[0]]

    @property
    def num_labels(self):
        return len(LABELS)
