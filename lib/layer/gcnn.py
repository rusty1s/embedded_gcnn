from six.moves import xrange

import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class GCNN(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 adj,
                 weight_stddev=0.1,
                 weight_decay=None,
                 bias=True,
                 bias_constant=0.1,
                 act=tf.nn.relu,
                 **kwargs):

        super(GCNN, self).__init__(**kwargs)

        self.adj = adj
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
        n = self.adj.get_shape()[0].value
        in_channels, out_channels = self.vars['weights'].get_shape()
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        # Align batches "horizontally", not "vertically".
        inputs = tf.transpose(inputs, [1, 0, 2])
        inputs = tf.reshape(inputs, [n, -1])

        # Multiply with adjacency matrix.
        outputs = tf.sparse_tensor_dense_matmul(self.adj, inputs)

        # Align output batches "vertically", not "horizontally".
        outputs = tf.reshape(outputs, [n, in_channels, -1])
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, in_channels])

        # Finally multiply with weight matrix.
        outputs = tf.matmul(outputs, self.vars['weights'])

        # Shape to 3D Tensor.
        outputs = tf.reshape(outputs, [-1, n, out_channels])

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
