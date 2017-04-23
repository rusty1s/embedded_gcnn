from __future__ import division

from six.moves import xrange

import tensorflow as tf
from numpy import pi as PI

from .var_layer import VarLayer


def base(adj_rad, K, P, p):
    assert p < P

    if K == 1:
        return None

    elif K == 2:
        shape = adj_rad.dense_shape
        zero = tf.SparseTensor(
            tf.constant(0, dtype=tf.int64, shape=[0, 2]),
            tf.constant(0, dtype=adj_rad.dtype, shape=[0]), shape)

        c = P / (2 * PI)  # Don't recalculate coefficient every time.
        values = adj_rad.values
        indices = adj_rad.indices

        left_values = c * values - p
        left = tf.SparseTensor(indices, left_values, shape)
        left = tf.sparse_maximum(left, zero)

        right_values = -c * values + p + 2
        right = tf.SparseTensor(indices, right_values, shape)
        right = tf.sparse_maximum(right, zero)

        offset_right_values = right_values - P
        offset_right = tf.SparseTensor(indices, offset_right_values, shape)
        offset_right = tf.sparse_maximum(offset_right, zero)

        return tf.sparse_maximum(tf.sparse_minimum(left, right), offset_right)
    else:
        raise NotImplementedError


def conv(features, adj_dist, adj_rad, weights, K):
    P = weights.get_shape()[0].value - 1

    output = tf.matmul(features, weights[P])

    for p in xrange(P):
        partition = base(adj_rad, K, P, p)

        # Note that we can perform elementwise multiplication on the two
        # adjacency matrices, although the sparse partition matrix has way less
        # elements than adj_dist. `base()` doesn't remove any element from
        # adj_rad and instead fills the irrelevant values with zeros. It is
        # nevertheless important that adj_dist and adj_rad have the same number
        # of elements with equal ordering.
        adj_values = tf.multiply(adj_dist.values, partition.values)
        adj = tf.SparseTensor(adj_dist.indices, adj_values,
                              adj_dist.dense_shape)

        output_p = tf.sparse_tensor_dense_matmul(adj, features)
        output_p = tf.matmul(output_p, weights[p])

        output += output_p

    return output


class EmbeddedGCNN(VarLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 adjs_dist,
                 adjs_rad,
                 local_controllability=2,
                 sampling_points=8,
                 **kwargs):

        assert local_controllability in [1, 2]

        self.adjs_dist = adjs_dist
        self.adjs_rad = adjs_rad
        self.K = local_controllability
        self.P = sampling_points

        super(EmbeddedGCNN, self).__init__(
            weight_shape=[self.P + 1, in_channels, out_channels],
            bias_shape=[out_channels],
            **kwargs)

    def _call(self, inputs):
        batch_size = inputs.get_shape[0].value
        outputs = []

        for i in xrange(batch_size):
            outputs.append(
                conv(inputs[i], self.adjs_dist[i], self.adjs_rad[i], self.K,
                     self.P, self.vars['weights']))

        outputs = tf.stack(outputs, axis=0)

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
