import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class VarLayer(Layer):
    def __init__(self,
                 weight_shape,
                 weight_stddev=0.1,
                 weight_decay=0.0,
                 bias=True,
                 bias_shape=[],
                 bias_constant=0.1,
                 act=tf.nn.relu,
                 **kwargs):

        super(VarLayer, self).__init__(**kwargs)

        self.act = act
        self.bias = bias

        with tf.variable_scope('{}_vars'.format(self.name)):
            self.vars['weights'] = weight_variable(
                weight_shape, '{}_weights'.format(self.name), weight_stddev,
                weight_decay)

            if self.bias:
                self.vars['bias'] = bias_variable(
                    bias_shape, '{}_bias'.format(self.name), bias_constant)

        if self.logging:
            self._log_vars()
