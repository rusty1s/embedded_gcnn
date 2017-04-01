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


class Cifar10(Datasets):
    def __init__(self, data_dir, validation_size=5000):
        maybe_download_and_extract(URL, data_dir)

        # Load the train data from disk.
        train_data = []
        train_labels = []
        for i in xrange(1, 6):
            data, labels = _load_batch(data_dir, 'data_batch_{}'.format(i))
            train_data.append(data)
            train_labels.append(labels)

        train_data = np.stack(train_data, axis=0)
        train_labels = np.stack(train_labels, axis=0)

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

    def _preprocess_all_data(self, images, preprocess, dataset):
        # HERE IS MORE TODO
        # CONVERT to float 0 bis 1
        return np.reshape(images, [-1, HEIGHT, WIDTH, DEPTH])


def _load_batch(data_dir, name):
    batch = pickle.load(open(os.path.join(data_dir, name)), 'rb')
    return batch.data, batch.labels
