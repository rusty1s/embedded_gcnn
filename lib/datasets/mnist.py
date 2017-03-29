import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from .dataset import DataSet


class MNIST(DataSet):
    def __init__(self, **kwargs):
        if 'data_dir' not in kwargs:
            kwargs['data_dir'] = 'data/mnist/input'

        super(MNIST, self).__init__(**kwargs)

        self.data = input_data.read_data_sets(self.data_dir, one_hot=False)

    def _reshape_images(self, images):
        return np.reshape(images, (images.shape[0], 28, 28, 1))

    @property
    def num_train_examples(self):
        return self.data.train.examples

    @property
    def num_validation_examples(self):
        return self.data.validation.examples

    @property
    def num_test_examples(self):
        return self.data.test.examples

    def next_train_batch(self, batch_size, shuffle=True):
        images, labels = self.data.train.next_batch(
            batch_size)
        return self._reshape_images(images), labels

    def next_validation_batch(self, batch_size, shuffle=True):
        images, labels = self.data.validation.next_batch(
            batch_size)
        return self._reshape_images(images), labels

    def next_test_batch(self, batch_size, shuffle=False):
        images, labels = self.data.test.next_batch(batch_size)
        return self._reshape_images(images), labels
