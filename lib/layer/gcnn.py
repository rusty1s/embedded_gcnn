from six.moves import xrange

import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class GCNN(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_stddev=0.1,
                 weight_decay=None,
                 bias=True,
                 bias_constant=0.1,
                 act=tf.nn.relu,
                 **kwargs):

        super(GCNN, self).__init__(**kwargs)

        self.bias = bias
        self.act = act

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

    def _call(self, inputs, adj):
        batch_size = inputs.get_shape()[0].value
        n = adj.get_shape()[1].value

        outputs = list()
        adj = tf.sparse_split(sp_input=adj, num_split=batch_size, axis=0)
        for i in xrange(batch_size):
            adj_i = tf.sparse_reshape(adj[i], [n, n])
            output = tf.sparse_tensor_dense_matmul(adj_i, inputs[i])
            output = tf.matmul(output, self.vars['weights'])

            outputs.append(output)

        outputs = tf.concat(outputs, axis=0)
        outputs = tf.reshape(outputs, [batch_size, n, -1])

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
