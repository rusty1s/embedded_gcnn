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
HEIGHT = 32
WIDTH = 32
DEPTH = 3

LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]


class Cifar10(Datasets):
    def __init__(self, data_dir, validation_size=5000):
        maybe_download_and_extract(URL, data_dir)
        data_dir = os.path.join(data_dir, 'cifar-10-batches-py')

        # Load the train data from disk.
        train_data = []
        train_labels = []
        for i in xrange(1, 6):
            data, labels = self._load_batch(data_dir,
                                            'data_batch_{}'.format(i))
            train_data.append(data)
            train_labels.append(labels)

        train_data = np.stack(train_data, axis=0)
        train_data = np.reshape(train_data, (-1, HEIGHT * WIDTH * DEPTH))
        train_labels = np.stack(train_labels, axis=0)
        train_labels = np.reshape(train_labels, (-1))

        train_data = train_data[validation_size:]
        train_labels = train_labels[validation_size:]
        train = Dataset(train_data, train_labels)

        # Generate a slice of validation data from trainings data.
        validation_data = train_data[:validation_size]
        validation_labels = train_labels[:validation_size]
        validation = Dataset(validation_data, validation_labels)

        # Load the test data from disk.
        test_data, test_labels = self._load_batch(data_dir, 'test_batch')
        test = Dataset(test_data, test_labels)

        super(Cifar10, self).__init__(train, validation, test, preprocess=True)

    def label_name(self, label):
        return [LABELS[i] for i in np.where(label == 1)[0]]

    @property
    def num_labels(self):
        return len(LABELS)

    def _preprocess_all(self, images, preprocess, dataset):
        images = np.reshape(images, (-1, DEPTH, HEIGHT, WIDTH))
        images = np.transpose(images, (0, 2, 3, 1))
        return (1 / 255) * images.astype(np.float32)

    def _load_batch(self, data_dir, name):
        batch = pickle.load(
            open(os.path.join(data_dir, name), 'rb'), encoding='latin1')

        labels = np.array(batch['labels'], np.uint8)
        index_offset = np.arange(labels.shape[0]) * self.num_labels
        labels_one_hot = np.zeros((labels.shape[0], self.num_labels), np.uint8)
        labels_one_hot.flat[index_offset + labels] = 1

        return batch['data'], labels_one_hot
