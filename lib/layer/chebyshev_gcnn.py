from six.moves import xrange

import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class ChebyshevGCNN(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 lap,
                 max_degree,
                 weight_stddev=0.1,
                 weight_decay=None,
                 bias=True,
                 bias_constant=0.1,
                 act=tf.nn.relu,
                 **kwargs):

        super(ChebyshevGCNN, self).__init__(**kwargs)

        self.lap = lap
        self.max_degree = max_degree
        self.bias = bias
        self.act = act

        with tf.variable_scope('{}_vars'.format(self.name)):
            self.vars['weights'] = weight_variable(
                [max_degree, in_channels, out_channels],
                '{}_weights'.format(self.name), weight_stddev, weight_decay)

            if self.bias:
                self.vars['bias'] = bias_variable(
                    [out_channels], '{}_bias'.format(self.name), bias_constant)

        if self.logging:
            self._log_vars()

    def _filter(self, Tx, degree):
        return tf.matmul(Tx, self.vars['weights'][degree])

    def _call(self, inputs):
        outputs = list()
        for i in xrange(inputs.get_shape()[0].value):
            Tx_0 = inputs[i]
            output = self._filter(Tx_0, 0)
            print(output)

            if self.max_degree > 0:
                Tx_1 = tf.sparse_tensor_dense_matmul(self.lap, inputs[i])
                output = tf.add(output, self._filter(Tx_1, 1))
                print(output)

            for k in xrange(2, self.max_degree + 1):
                print(k)
                Tx_2 = 2 * tf.sparse_tensor_dense_matmul(self.lap, Tx_1) - Tx_0
                output += tf.add(output, self._filter(Tx_2, k))
                print(output)

                Tx_0, Tx_1 = Tx_1, Tx_2

            outputs.append(output)

        outputs = tf.stack(outputs, axis=0)

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
