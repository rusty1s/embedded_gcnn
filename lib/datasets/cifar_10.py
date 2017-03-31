import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from .dataset import Datasets, Dataset


class MNIST(Datasets):
    def __init__(self, data_dir):
        mnist = input_data.read_data_sets(data_dir)

        train = Dataset(mnist.train.images, mnist.train.labels)
        validation = Dataset(mnist.validation.images, mnist.validation.labels)
        test = Dataset(mnist.test.images, mnist.test.labels)

        super(MNIST, self).__init__(train, validation, test, preprocess=True)

    def _preprocess_all_data(self, images, preprocess, dataset):
        return np.reshape(images, [-1, 28, 28, 1])
