import tensorflow as tf

from .layer import Layer


class MaxPool(Layer):
    def __init__(self, size, stride=None, **kwargs):
        self.size = size
        self.stride = size if stride is None else stride

        super(MaxPool, self).__init__(**kwargs)

    def _call(self, inputs):
        rank = len(inputs.get_shape())

        if rank == 3:
            # Use 1D Pooling.
            outputs = tf.expand_dims(inputs, axis=3)
            outputs = tf.nn.max_pool(
                outputs,
                ksize=[1, self.size, 1, 1],
                strides=[1, self.stride, 1, 1],
                padding='SAME')
            return tf.squeeze(outputs, axis=3)

        elif rank == 4:
            # Use 2D Pooling.
            return tf.nn.max_pool(
                inputs,
                ksize=[1, self.size, self.size, 1],
                strides=[1, self.stride, self.stride, 1],
                padding='SAME')

        else:
            raise AssertionError
