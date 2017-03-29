# from six.moves import xrange

import tensorflow as tf

from .layer import Layer


class MaxPoolGCNN(Layer):
    def __init__(self, size, **kwargs):

        super(MaxPoolGCNN, self).__init__(**kwargs)

        self.size = size

    def _call(self, inputs):
        outputs = tf.expand_dims(inputs, axis=3)
        outputs = tf.nn.max_pool(
            outputs,
            ksize=[1, self.size, 1, 1],
            strides=[1, self.size, 1, 1],
            padding='SAME')
        return tf.squeeze(outputs, axis=3)
