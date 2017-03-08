from six.moves import xrange

import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class ChebyshevGCNN(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 max_degree,
                 weight_stddev=0.1,
                 weight_decay=None,
                 bias=True,
                 bias_constant=0.1,
                 act=tf.nn.relu,
                 **kwargs):

        super(ChebyshevGCNN, self).__init__(**kwargs)

        self.max_degree = max_degree
        self.bias = bias
        self.act = act

        with tf.variable_scope('{}_vars'.format(self.name)):
            self.vars['weights'] = weight_variable(
                [max_degree, in_channels, out_channels],
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

    def _filter(self, cheb, degree):
        weights = self.vars['weights'][degree]
        return tf.matmul(cheb, weights)

    def _call(self, inputs, lap):
        batch_size = inputs.get_shape()[0].value
        n = lap.get_shape()[1].value

        outputs = list()
        lap = tf.sparse_split(sp_input=lap, num_split=batch_size, axis=0)
        for i in xrange(batch_size):
            lap_i = tf.sparse_reshape(lap[i], [n, n])

            cheb_0 = inputs[i]
            output = self._filter(cheb_0, 0)

            if self.max_degree > 0:
                cheb_1 = tf.sparse_tensor_dense_matmul(lap_i, inputs[i])
                output += self._filter(cheb_1, 1)

            for k in xrange(2, self.max_degree + 1):
                cheb_2 = 2 * tf.sparse_tensor_dense_matmul(lap_i,
                                                           cheb_1) - cheb_0
                output += self._filter(cheb_2, k)
                cheb_0, cheb_1 = cheb_1, cheb_2

            outputs.append(output)

        outputs = tf.concat(outputs, axis=0)
        outputs = tf.reshape(outputs, [batch_size, n, -1])

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
