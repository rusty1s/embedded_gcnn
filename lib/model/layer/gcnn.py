import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


def diag(values):
    n = values.get_shape()[0].value
    i = tf.reshape(tf.range(0, n), [n, 1])
    indices = tf.concat([i, i], axis=1)
    return tf.SparseTensor(tf.cast(indices, tf.int64), values, [n, n])


def normalize_adj(adj, dtype=tf.float32):
    adj = tf.cast(adj, dtype)
    n = adj.get_shape()[0].value
    return tf.sparse_add(adj, diag(tf.ones([n], dtype)))


class GraphConvolution(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_stddev,
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

    def _call(self, inputs, adjacencies):

        # normaliez adjacencies
        # calculate degree
        # inv_sqrt

        outputs = tf.m
        outputs = inputs

        if self.bias:
            outputs += self.vars['bias']

        return self.act(outputs)
