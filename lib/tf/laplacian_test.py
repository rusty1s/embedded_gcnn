from numpy.testing import assert_almost_equal
import scipy.sparse as sp
import tensorflow as tf

from .laplacian import laplacian, rescale_lap
from .convert import sparse_to_tensor


class LaplacianTest(tf.test.TestCase):
    def test_laplacian(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)
        adj_tf = sparse_to_tensor(adj)
        lap = laplacian(adj_tf)
        lap = tf.sparse_tensor_to_dense(lap)

        expected = sp.csgraph.laplacian(adj, normed=True).toarray()

        with self.test_session():
            assert_almost_equal(lap.eval(), expected)

    def test_rescale_lap(self):
        lap = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        lap = sp.coo_matrix(lap)
        lap = sparse_to_tensor(lap)
        lap = rescale_lap(lap)
        lap = tf.sparse_tensor_to_dense(lap)

        expected = [[-1, 1, 0], [1, -1, 2], [0, 2, -1]]

        with self.test_session():
            self.assertAllEqual(lap.eval(), expected)
