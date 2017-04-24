import numpy as np
from numpy.testing import assert_almost_equal
import scipy.sparse as sp
import tensorflow as tf

from .adjacency import normalize_adj
from .convert import sparse_to_tensor


class AdjacencyTest(tf.test.TestCase):
    def test_normalize_adj(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)
        adj_tf = sparse_to_tensor(adj)
        adj_norm = normalize_adj(adj_tf)
        adj_norm = tf.sparse_tensor_to_dense(adj_norm)

        degree = np.array(adj.sum(axis=1)).flatten()
        degree = np.power(degree, -0.5)
        degree = sp.diags(degree)
        expected = degree.dot(adj).dot(degree).toarray()

        with self.test_session():
            assert_almost_equal(adj_norm.eval(), expected)
