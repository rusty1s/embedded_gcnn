from __future__ import division

import tensorflow as tf

from .layer import Layer


class MaxPoolGCNN(Layer):
    def __init__(self, size, **kwargs):

        super(MaxPoolGCNN, self).__init__(**kwargs)

        self.size = size

    def _call(self, inputs):
        n = inputs.get_shape()[1].value
        in_channels = inputs.get_shape()[2].value

        inputs = tf.reshape(inputs, [-1, 1, n, in_channels])

        inputs = tf.nn.max_pool(
            inputs,
            ksize=[1, 1, self.size, 1],
            strides=[1, 1, self.size, 1],
            padding='SAME')

        return tf.reshape(inputs, [-1, n // self.size, in_channels])
