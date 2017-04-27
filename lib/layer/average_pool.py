from six.moves import xrange

import tensorflow as tf

from .layer import Layer


class AveragePool(Layer):
    def __init__(self, **kwargs):
        super(AveragePool, self).__init__(**kwargs)

    def _call(self, inputs):
        batch_size = len(inputs)
        outputs = []
        for i in xrange(batch_size):
            output = tf.reduce_mean(inputs[i], axis=0)
            outputs.append(output)
        return tf.stack(outputs)
