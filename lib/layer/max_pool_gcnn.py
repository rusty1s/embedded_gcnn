from six.moves import xrange

import tensorflow as tf

from .layer import Layer


class MaxPoolGCNN(Layer):
    def __init__(self, size, **kwargs):

        super(MaxPoolGCNN, self).__init__(**kwargs)

        self.size = size

    def _call(self, inputs):
        multiple = isinstance(inputs, list)

        if multiple:
            batch_size = len(inputs)

            outputs = []
            for i in xrange(batch_size):
                output = tf.expand_dims(inputs[i], axis=0)
                output = tf.expand_dims(output, axis=3)
                output = tf.nn.max_pool(
                    output,
                    ksize=[1, self.size, 1, 1],
                    strides=[1, self.size, 1, 1],
                    padding='SAME')
                output = tf.squeeze(output, axis=[0, 3])
                outputs.append(output)
            return outputs

        else:
            outputs = tf.expand_dims(inputs, axis=3)
            outputs = tf.nn.max_pool(
                outputs,
                ksize=[1, self.size, 1, 1],
                strides=[1, self.size, 1, 1],
                padding='SAME')
            return tf.squeeze(outputs, axis=3)
