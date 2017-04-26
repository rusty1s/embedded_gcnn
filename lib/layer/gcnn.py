from six.moves import xrange

import tensorflow as tf

from .var_layer import VarLayer
from ..tf.adjacency import normalize_adj
from ..tf.math import sparse_identity


def conv(features, adj, weights):
    N = adj.dense_shape[0]

    adj = tf.sparse_add(adj, sparse_identity(N, adj.values.dtype))
    adj = normalize_adj(adj)

    output = tf.sparse_tensor_dense_matmul(adj, features)
    return tf.matmul(output, weights)


class GCNN(VarLayer):
    def __init__(self, in_channels, out_channels, adjs, **kwargs):
        self.adjs = adjs

        super(GCNN, self).__init__(
            weight_shape=[in_channels, out_channels],
            bias_shape=[out_channels],
            **kwargs)

    def _call(self, inputs):
        batch_size = inputs.get_shape()[0].value
        outputs = []

        for i in xrange(batch_size):
            outputs.append(conv(inputs[i], self.adjs[i], self.vars['weights']))

        outputs = tf.stack(outputs, axis=0)

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
