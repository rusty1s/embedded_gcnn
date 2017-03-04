import tensorflow as tf
import numpy as np


class gcnn(object):
    def __init__(self, L, k, num_filters, normalized=True):
        super().__init__()
        self.L = L
        self.k = k
        self.num_filters = num_filters
        self.normalized = normalized

    def _inference(self, X):
        num_samples, num_features = X.get_shape()
