from six.moves import xrange

import tensorflow as tf

from .var_layer import VarLayer
from ..tf.bspline import base


def conv(features, adj_dist, adj_rad, K, weights):
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
        batch_size = len(inputs)
        outputs = []

        for i in xrange(batch_size):
            output = conv(inputs[i], self.adjs_dist[i], self.adjs_rad[i],
                          self.K, self.vars['weights'])

            if self.bias:
                output = tf.nn.bias_add(output, self.vars['bias'])

            output = self.act(output)
            outputs.append(output)

        return outputs
