import tensorflow as tf

from .var_layer import VarLayer


def conv(inputs, weights):
    outputs = tf.nn.conv2d(
        inputs,
        weights,
        strides=[1, 1, weights.get_shape()[1], 1],
        padding='VALID')
    return tf.squeeze(outputs, axis=2)


class SpatialCNN(VarLayer):
    def __init__(self, in_channels, out_channels, neighborhood_size, **kwargs):

        super(SpatialCNN, self).__init__(
            weight_shape=[1, neighborhood_size, in_channels, out_channels],
            bias_shape=[out_channels],
            **kwargs)

    def _call(self, inputs):
        outputs = conv(inputs, self.vars['weights'])

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
