import numpy as np
import scipy.sparse as sp
import tensorflow as tf


class SparseTest(tf.test.TestCase):
    def test_mul(self):
        dense_adj = np.array(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        dense_adj_tf = tf.constant(dense_adj)

        sparse_adj = sp.coo_matrix(dense_adj)
        indices = np.concatenate(
            (np.reshape(sparse_adj.row, (-1, 1)), np.reshape(sparse_adj.col,
                                                             (-1, 1))),
            axis=1)
        sparse_adj_tf = tf.SparseTensor(indices, sparse_adj.data,
                                        sparse_adj.shape)

        features = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
        features_tf = tf.constant(features)

        multiple_features = np.concatenate((features, features), axis=0)
        multiple_features = np.reshape(multiple_features, (2, 3, 2))
        multiple_features_tf = tf.constant(multiple_features)
