from six.moves import xrange

import numpy as np
import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class GCNN(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 adjs,
                 weight_stddev=0.1,
                 weight_decay=None,
                 bias=False,
                 bias_constant=0.1,
                 act=tf.nn.relu,
                 **kwargs):

        super(GCNN, self).__init__(**kwargs)

        self.adjs = adjs
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
        n = inputs.get_shape()[1].value
        in_channels = inputs.get_shape()[2].value

        # We add a zero to the output, so TensowFlow knows the shape of the
        # bugged sparse placehoder shape. This is not elegant, but there's
        # no other way :(
        zero = tf.zeros([n, in_channels], dtype=inputs.dtype)

        multiple = isinstance(self.adjs, (list, tuple))

        outputs = list()
        for i in xrange(inputs.get_shape()[0].value):
            adj = self.adjs[i] if multiple else self.adjs

            output = tf.sparse_tensor_dense_matmul(adj, inputs[i])
            output = tf.add(output, zero)

            output = tf.matmul(output, self.vars['weights'])
            outputs.append(output)

        outputs = tf.stack(outputs, axis=0)

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
