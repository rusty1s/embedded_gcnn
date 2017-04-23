import tensorflow as tf

from .layer import Layer


class MaxPool2d(Layer):
    def __init__(self, size, stride, **kwargs):

        super(MaxPool2d, self).__init__(**kwargs)

        self.size = [1, size, size, 1]
        self.stride = [1, stride, stride, 1]

    def _call(self, inputs):
        return tf.nn.max_pool(
            inputs, ksize=self.size, strides=self.stride, padding='SAME')
