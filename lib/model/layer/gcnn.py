from six.moves import xrange

import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class GCNN(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_stddev=0.0,
                 weight_decay=None,
                 bias=False,
                 bias_constant=0.0,
                 act=tf.nn.relu,
                 **kwargs):

        super().__init__(**kwargs)

        self.act = act
        self.bias = bias

        with tf.variable_scope('{}_vars'.format(self.name)):
            self.vars['weights'] = weight_variable(
                [in_channels, out_channels],
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

    def _call(self, inputs, A):
        batch_size = inputs.get_shape()[0].value
        n = A.get_shape()[1].value

        outputs = list()
        A = tf.sparse_split(sp_input=A, num_split=batch_size, axis=0)
        for i in xrange(batch_size):
            a = A[i]
            a = tf.sparse_reshape(a, [n, n])
            output = tf.sparse_tensor_dense_matmul(a, inputs[i])
            output = tf.matmul(output, self.vars['weights'])

            if self.bias:
                output += self.vars['bias']

            outputs.append(output)

        outputs = tf.concat(outputs, axis=0)
        outputs = tf.reshape(outputs, [batch_size, n, -1])
        return self.act(outputs)
