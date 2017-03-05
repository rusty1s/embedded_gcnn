from six.moves import xrange

import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class ChebyshevGCNN(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 max_degree,
                 weight_stddev=0.01,
                 weight_decay=None,
                 bias=True,
                 bias_constant=0.0,
                 act=tf.nn.relu,
                 **kwargs):

        super().__init__(**kwargs)

        self.max_degree = max_degree
        self.bias = bias
        self.act = act

        with tf.variable_scope('{}_vars'.format(self.name)):
            self.vars['weights'] = weight_variable(
                [max_degree, in_channels, out_channels],
                weight_stddev,
                weight_decay,
                name='{}_weights'.format(self.name))

            if self.bias:
                self.vars['bias'] = bias_variable(
                    [out_channels],
                    bias_constant,
                    name='{}_bias'.format(self.name))

        if self.logging:
            self._log_vars()

    def _filter(self, T, k):
        W = self.vars['weights'][k]
        return tf.matmul(T, W)

    def _call(self, inputs, L):
        batch_size = inputs.get_shape()[0].value
        n = L.get_shape()[1].value

        outputs = list()
        L = tf.sparse_split(sp_input=L, num_split=batch_size, axis=0)
        for i in xrange(batch_size):
            L_i = tf.sparse_reshape(L[i], [n, n])

            T_0 = inputs[i]
            output = self._filter(T_0, 0)

            if self.k > 0:
                T_1 = tf.sparse_tensor_dense_matmul(L_i, inputs[i])
                output += self._filter(T_1, 1)

            for k in xrange(2, self.max_degree + 1):
                T_2 = 2 * tf.sparse_tensor_dense_matmul(L_i, T_1) - T_0
                output += self._filter(T_2, k)
                T_0, T_1 = T_1, T_2

            outputs.append(output)

        outputs = tf.concat(outputs, axis=0)
        outputs = tf.reshape(outputs, [batch_size, n, -1])

        if self.bias:
            outputs += self.vars['bias']

        return self.act(outputs)
