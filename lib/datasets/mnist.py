import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from .dataset import Datasets, Dataset

HEIGHT = 28
WIDTH = 28
DEPTH = 1


class MNIST(Datasets):
    def __init__(self, data_dir):
        mnist = input_data.read_data_sets(data_dir, one_hot=True)

        train = Dataset(mnist.train.images, mnist.train.labels)
        validation = Dataset(mnist.validation.images, mnist.validation.labels)
        test = Dataset(mnist.test.images, mnist.test.labels)

        super(MNIST, self).__init__(train, validation, test, preprocess=True)

    def label_name(self, label):
        return np.where(label == 1)[0].tolist()

    @property
    def num_labels(self):
        return 10

    def _preprocess_all(self, images, preprocess, dataset):
        return np.reshape(images, [-1, HEIGHT, WIDTH, DEPTH])
