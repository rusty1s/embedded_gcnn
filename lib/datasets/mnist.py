import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from .dataset import Datasets, Dataset


def _preprocess_images(images):
    return np.reshape(images, (-1, 28, 28, 1))


def _preprocess_labels(labels):
    return labels.astype(np.uint8)


class MNIST(Datasets):
    def __init__(self, data_dir, val_size=5000):
        mnist = input_data.read_data_sets(
            data_dir, one_hot=True, validation_size=val_size)

        images = _preprocess_images(mnist.train.images)
        labels = _preprocess_labels(mnist.train.labels)
        train = Dataset(images, labels)

        images = _preprocess_images(mnist.validation.images)
        labels = _preprocess_labels(mnist.validation.labels)
        val = Dataset(images, labels)

        images = _preprocess_images(mnist.test.images)
        labels = _preprocess_labels(mnist.test.labels)
        test = Dataset(images, labels)

        super(MNIST, self).__init__(train, val, test)

    @property
    def classes(self):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
