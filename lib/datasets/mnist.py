import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from .dataset import Datasets, Dataset


class MNIST(Datasets):
    def __init__(self, data_dir, val_size=5000):
        mnist = input_data.read_data_sets(
            data_dir, one_hot=True, validation_size=val_size)

        images = self._preprocess_images(mnist.train.images)
        labels = self._preprocess_labels(mnist.train.labels)
        train = Dataset(images, labels)

        images = self._preprocess_images(mnist.validation.images)
        labels = self._preprocess_labels(mnist.validation.labels)
        val = Dataset(images, labels)

        images = self._preprocess_images(mnist.test.images)
        labels = self._preprocess_labels(mnist.test.labels)
        test = Dataset(images, labels)

        super(MNIST, self).__init__(train, val, test)

    @property
    def classes(self):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    @property
    def width(self):
        return 28

    @property
    def height(self):
        return 28

    @property
    def num_channels(self):
        return 1

    def _preprocess_images(self, images):
        return np.reshape(images, (-1, self.height, self.width,
                                   self.num_channels))

    def _preprocess_labels(self, labels):
        return labels.astype(np.uint8)
