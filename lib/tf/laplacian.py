import tensorflow as tf

from .math import sparse_identity, sparse_subtract
from .adjacency import normalize_adj


def laplacian(adj, dtype=tf.float32):
    adj_norm = normalize_adj(adj, dtype)

    n = adj.dense_shape[0]
    identity = sparse_identity(n, dtype)

    return sparse_subtract(identity, adj_norm)


def rescale_lap(lap):
    n = lap.dense_shape[0]
    identity = sparse_identity(n, lap.values.dtype)
    return sparse_subtract(lap, identity)
