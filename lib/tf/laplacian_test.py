from numpy.testing import assert_almost_equal
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian
import tensorflow as tf

from .laplacian import rescaled_laplacian
from .convert import sparse_to_tensor


class LaplacianTest(tf.test.TestCase):
    def test_rescaled_laplacian(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)
        adj_tf = sparse_to_tensor(adj)
        lap = rescaled_laplacian(adj_tf)
        lap = tf.sparse_tensor_to_dense(lap)

        expected = laplacian(adj, normed=True) - sp.eye(3)
        expected = expected.toarray()

        with self.test_session():
            assert_almost_equal(lap.eval(), expected)
