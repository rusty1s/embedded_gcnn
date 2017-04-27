import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from .dataset import Datasets, Dataset


def _preprocess(images):
    return np.reshape(images, (-1, 28, 28, 1))


class MNIST(Datasets):
    def __init__(self, data_dir):
        mnist = input_data.read_data_sets(data_dir, one_hot=True)

        train = Dataset(_preprocess(mnist.train.images), mnist.train.labels)
        validation = Dataset(
            _preprocess(mnist.validation.images), mnist.validation.labels)
        test = Dataset(_preprocess(mnist.test.images), mnist.test.labels)

        super(MNIST, self).__init__(train, validation, test)

    def label_name(self, label):
        return np.where(label == 1)[0].tolist()

    @property
    def num_labels(self):
        return 10
