import tensorflow as tf

from .var_layer import VarLayer


def conv(inputs, weights, stride):
    return tf.nn.conv2d(
        inputs, weights, [1, stride, stride, 1], padding='SAME')


class Conv2d(VarLayer):
    def __init__(self, in_channels, out_channels, size=3, stride=1, **kwargs):
        self.stride = stride

        super(Conv2d, self).__init__(
            weight_shape=[size, size, in_channels, out_channels],
            bias_shape=[out_channels],
            **kwargs)

    def _call(self, inputs):
        outputs = conv(inputs, self.vars['weights'], self.stride)

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return outputs
