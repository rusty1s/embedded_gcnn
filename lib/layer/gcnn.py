from six.moves import xrange

import tensorflow as tf

from .var_layer import VarLayer
from ..tf import sparse_identity, normalize_adj


def conv(features, adj, weights):
    n = adj.dense_shape[0]

    adj = tf.sparse_add(adj, sparse_identity(n, adj.values.dtype))
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
        batch_size = len(inputs)
        outputs = []

        for i in xrange(batch_size):
            output = conv(inputs[i], self.adjs[i], self.vars['weights'])

            if self.bias:
                output = tf.nn.bias_add(output, self.vars['bias'])

            output = self.act(output)
            outputs.append(output)

        return outputs
