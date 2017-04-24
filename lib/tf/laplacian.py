import tensorflow as tf

from .math import sparse_identity, sparse_subtract
from .adjacency import normalize_adj


def laplacian(adj, dtype=tf.float32):
    adj_norm = normalize_adj(adj, dtype)

    N = adj.dense_shape[0]
    I = sparse_identity(N, dtype)

    return sparse_subtract(I, adj_norm)


def rescale_lap(lap):
    N = lap.dense_shape[0]
    I = sparse_identity(N, lap.values.dtype)
    return sparse_subtract(lap, I)
