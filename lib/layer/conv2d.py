import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class Conv2d(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 size,
                 stride,
                 weight_stddev=0.1,
                 weight_decay=None,
                 bias=True,
                 bias_constant=0.1,
                 act=tf.nn.relu,
                 **kwargs):

        super(Conv2d, self).__init__(**kwargs)

        self.stride = stride
        self.bias = bias
        self.act = act

        with tf.variable_scope('{}_vars'.format(self.name)):
            self.vars['weights'] = weight_variable(
                [size, size, in_channels, out_channels],
                '{}_weights'.format(self.name),
                weight_stddev,
                weight_decay)

            if self.bias:
                self.vars['bias'] = bias_variable(
                    [out_channels],
                    '{}_bias'.format(self.name),
                    bias_constant)

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        outputs = tf.nn.conv2d(
            inputs,
            self.vars['weights'], [1, self.stride, self.stride, 1],
            padding='SAME')

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
