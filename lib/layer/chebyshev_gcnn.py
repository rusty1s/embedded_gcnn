from six.moves import xrange

import tensorflow as tf

from .var_layer import VarLayer
from ..tf.math import sparse_identity, sparse_subtract


def conv(features, lap, weights):
    K = weights.get_shape()[0].value - 1
    N = features.get_shape()[0].value

    # Rescale normalized laplacian.
    lap = sparse_subtract(lap, sparse_identity(N, lap.dtype))

    Tx_0 = features
    output = tf.matmul(Tx_0, weights[0])

    if K > 0:
        Tx_1 = tf.sparse_tensor_dense_matmul(lap, features)
        output += tf.matmul(Tx_1, weights[1])

    for k in xrange(2, K + 1):
        Tx_2 = 2 * tf.sparse_tensor_dense_matmul(lap, Tx_1) - Tx_0
        output += tf.matmul(Tx_2, k)

        Tx_0, Tx_1 = Tx_1, Tx_2

    return output


class ChebyshevGCNN(VarLayer):
    def __init__(self, in_channels, out_channels, laps, degree,
                 **kwargs):

        self.laps = laps

        super(ChebyshevGCNN, self).__init__(
            weight_shape=[degree + 1, in_channels, out_channels],
            bias_shape=[out_channels],
            **kwargs)

    def _call(self, inputs):
        batch_size = inputs.get_shape()[0].value
        outputs = []

        for i in xrange(batch_size):
            outputs.append(conv(inputs[i], self.laps[i], self.vars['weights']))

        outputs = tf.stack(outputs, axis=0)

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
