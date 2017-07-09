from six.moves import xrange

import tensorflow as tf

from .var_layer import VarLayer
from ..tf import sparse_identity, sparse_tensor_diag_matmul


def conv(features, adj, weights):
    n = adj.dense_shape[0]

    degree = tf.sparse_reduce_sum(adj, axis=1) + 1
    degree = tf.cast(degree, tf.float32)
    degree = tf.pow(degree, -0.5)

    adj = sparse_tensor_diag_matmul(adj, degree, transpose=True)
    adj = sparse_tensor_diag_matmul(adj, degree, transpose=False)

    output = tf.sparse_tensor_dense_matmul(adj, features)

    features = tf.transpose(features)
    features = tf.multiply(tf.multiply(degree, features), degree)
    features = tf.transpose(features)
    output = output + features

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
