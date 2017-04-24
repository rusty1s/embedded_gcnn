import tensorflow as tf

from .layer import Layer


class AveragePool(Layer):
    def __init__(self, **kwargs):
        super(AveragePool, self).__init__(**kwargs)

    def _call(self, inputs):
        rank = len(inputs.get_shape())

        if rank == 3:
            return tf.reduce_mean(inputs, axis=1)

        elif rank == 4:
            # Reshape to Rank 3 Tensor.
            _, height, width, depth = inputs.get_shape()
            outputs = tf.reshape(inputs,
                                 [-1, int(height) * int(width), int(depth)])
            return tf.reduce_mean(outputs, axis=1)

        else:
            raise AssertionError
