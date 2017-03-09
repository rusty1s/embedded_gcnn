import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class FC(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.0,
                 placeholders={},
                 weight_stddev=0.1,
                 weight_decay=None,
                 bias=True,
                 bias_constant=0.1,
                 act=tf.nn.relu,
                 **kwargs):

        super(FC, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.dropout = dropout
        self.bias = bias
        self.act = act

        with tf.variable_scope('{}_vars'.format(self.name)):
            self.vars['weights'] = weight_variable(
                [in_channels, out_channels], '{}_weights'.format(self.name),
                weight_stddev, weight_decay)

            if self.bias:
                self.vars['bias'] = bias_variable(
                    [out_channels], '{}_bias'.format(self.name), bias_constant)

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        outputs = tf.reshape(inputs, [-1, self.in_channels])

        outputs = tf.nn.dropout(outputs, 1 - self.dropout)
        outputs = tf.matmul(outputs, self.vars['weights'])

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
