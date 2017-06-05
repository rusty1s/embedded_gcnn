from six.moves import xrange

import tensorflow as tf

from .layer import Layer


class MaxPool(Layer):
    def __init__(self, size, stride=None, **kwargs):
        self.size = size
        self.stride = size if stride is None else stride

        super(MaxPool, self).__init__(**kwargs)

    def _call(self, inputs):
        # Different treatment of the inputs, whether we learn on arbitary
        # graphs or images. Graph features are passed as a normal python list
        # and need a special adjustment, whereas image tensors use the normal
        # tensorflow `max_pool` operation.

        if isinstance(inputs, list):
            batch_size = len(inputs)

            outputs = []
            for i in xrange(batch_size):
                output = tf.expand_dims(inputs[i], axis=0)
                output = tf.expand_dims(output, axis=3)
                output = tf.nn.max_pool(
                    output,
                    ksize=[1, self.size, 1, 1],
                    strides=[1, self.stride, 1, 1],
                    padding='SAME')
                output = tf.squeeze(output, axis=[0, 3])
                outputs.append(output)

            return outputs
        else:
            return tf.nn.max_pool(
                inputs,
                ksize=[1, self.size, self.size, 1],
                strides=[1, self.stride, self.stride, 1],
                padding='SAME')
