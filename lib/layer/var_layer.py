import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class VarLayer(Layer):
    def __init__(self,
                 weight_shape,
                 bias_shape,
                 weight_stddev=0.1,
                 weight_decay=0.0,
                 bias=True,
                 bias_constant=0.1,
                 bias_decay=0.0,
                 act=tf.nn.relu,
                 **kwargs):

        super(VarLayer, self).__init__(**kwargs)

        self.bias = bias
        self.act = act
        self.vars = {}

        with tf.variable_scope('{}_vars'.format(self.name)):
            self.vars['weights'] = weight_variable(
                weight_shape, '{}_weights'.format(self.name), weight_stddev,
                weight_decay)

            if self.bias:
                self.vars['bias'] = bias_variable(bias_shape,
                                                  '{}_bias'.format(self.name),
                                                  bias_constant, bias_decay)

        if self.logging:
            for var in self.vars:
                tf.summary.histogram('{}/vars/{}'.format(self.name, var),
                                     self.vars[var])
