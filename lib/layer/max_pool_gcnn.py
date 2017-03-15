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


# x = tf.expand_dims(x, 3)  # N x M x F x 1
# x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
# #tf.maximum
# return tf.squeeze(x, [3])  # N x M/p x F
