import tensorflow as tf

from .math import sparse_tensor_diag_matmul


def normalize_adj(adj):
    diag = tf.sparse_reduce_sum(adj, axis=1)
    diag = tf.pow(diag, -0.5)
    adj_norm = sparse_tensor_diag_matmul(adj, diag, transpose=True)
    adj_norm = sparse_tensor_diag_matmul(adj_norm, diag, transpose=False)
    return adj_norm
