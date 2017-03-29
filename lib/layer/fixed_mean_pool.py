from six.moves import xrange

import tensorflow as tf

from .layer import Layer


class FixedMeanPool(Layer):
    def __init__(self, **kwargs):
        super(FixedMeanPool, self).__init__(**kwargs)

    def _call(self, inputs):
        multiple = isinstance(inputs, list)

        if multiple:
            x = tf.zeros((inputs[0].get_shape()[1].value), inputs[0].dtype)
            batch_size = len(inputs)
            outputs = []
            for i in xrange(batch_size):
                outputs.append(tf.reduce_mean(inputs[i], axis=0))
            x = x + tf.stack(outputs)
            return x
        else:
            if len(list(inputs.get_shape())) == 4:
                inputs = tf.reshape(inputs, [
                    -1, inputs.get_shape()[1].value *
                    inputs.get_shape()[2].value, inputs.get_shape()[3].value
                ])

            return tf.reduce_mean(inputs, axis=1)
