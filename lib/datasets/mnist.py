import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from .dataset import DataSet


class MNIST(DataSet):
    def __init__(self, **kwargs):
        if 'data_dir' not in kwargs:
            kwargs['data_dir'] = 'data/mnist/input'

        super(MNIST, self).__init__(**kwargs)

        self.data = input_data.read_data_sets(self.data_dir, one_hot=False)

        self.height = 28
        self.width = 28
        self.channels = 1
        self.num_train_examples = self.data.train.images.shape[0]
        self.num_validation_examples = self.data.validation.images.shape[0]
        self.num_test_examples = self.data.test.images.shape[0]

    def _reshape_images(self, images):
        return np.reshape(images, (images.shape[0], self.height, self.width,
                                   self.channels))

    def next_train_batch(self, batch_size):
        images, labels = self.data.train.next_batch(batch_size)
        return self._reshape_images(images), labels

    def next_validation_batch(self, batch_size):
        images, labels = self.data.validation.next_batch(batch_size)
        return self._reshape_images(images), labels

    def next_test_batch(self, batch_size):
        images, labels = self.data.test.next_batch(batch_size)
        return self._reshape_images(images), labels
