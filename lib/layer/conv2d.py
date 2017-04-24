import tensorflow as tf

from .var_layer import VarLayer


class Conv2d(VarLayer):
    def __init__(self, in_channels, out_channels, size, stride, **kwargs):
        self.stride = [1, stride, stride, 1]

        super(Conv2d, self).__init__(
            weight_shape=[size, size, in_channels, out_channels],
            bias_shape=[out_channels],
            **kwargs)

    def _call(self, inputs):
        outputs = tf.nn.conv2d(
            inputs, self.vars['weights'], self.stride, padding='SAME')

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
