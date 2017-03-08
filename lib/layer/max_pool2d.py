import tensorflow as tf

from .layer import Layer


class MaxPool2d(Layer):
    def __init__(self, size, stride, **kwargs):

        super(MaxPool2d, self).__init__(**kwargs)

        self.size = size
        self.stride = stride

    def _call(self, inputs):
        return tf.nn.max_pool(
            inputs,
            ksize=[1, self.size, self.size, 1],
            strides=[1, self.stride, self.stride, 1],
            padding='SAME')
