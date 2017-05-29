import tensorflow as tf

from .math import sparse_identity, sparse_subtract, sparse_tensor_diag_matmul


def laplacian(adj):
    degree = tf.sparse_reduce_sum(adj, axis=1)
    degree = tf.cast(degree, tf.float32)
    degree = tf.pow(degree, -0.5)
    adj = sparse_tensor_diag_matmul(adj, degree, transpose=True)
    adj = sparse_tensor_diag_matmul(adj, degree, transpose=False)

    n = adj.dense_shape[0]
    identity = sparse_identity(n, tf.float32)

    return sparse_subtract(identity, adj)


def rescale_lap(lap):
    n = lap.dense_shape[0]
    identity = sparse_identity(n, lap.values.dtype)
    return sparse_subtract(lap, identity)
