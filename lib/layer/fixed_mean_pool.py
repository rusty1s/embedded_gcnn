from six.moves import xrange

import tensorflow as tf

from .layer import Layer


class FixedMeanPool(Layer):
    def __init__(self, **kwargs):
        super(FixedMeanPool, self).__init__(**kwargs)

    def _call(self, inputs):
        multiple = isinstance(inputs, list)

        if multiple:
            batch_size = len(inputs)
            outputs = []
            for i in xrange(batch_size):
                outputs.append(tf.reduce_mean(inputs[i], axis=0))
            return tf.stack(outputs)
        else:
            return tf.reduce_mean(inputs, axis=1)
