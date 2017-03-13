import tensorflow as tf
import scipy.sparse as sp

from .sparse import sparse_to_tensor


class SparseTest(tf.test.TestCase):
    def test_sparse_to_tensor(self):
        value = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        value = sp.coo_matrix(value)

        with self.test_session():
            self.assertAllEqual(
                tf.sparse_tensor_to_dense(sparse_to_tensor(value)).eval(),
                value.toarray())
