import scipy.sparse as sp
import tensorflow as tf

from .math import (sparse_identity, sparse_scalar_multiply, sparse_subtract,
                   sparse_tensor_diag_matmul)
from .convert import sparse_to_tensor


class MathTest(tf.test.TestCase):
    def test_sparse_identity(self):
        I = sparse_identity(3)
        I = tf.sparse_tensor_to_dense(I)
        expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        with self.test_session():
            self.assertAllEqual(I.eval(), expected)

    def test_sparse_scalar_multiply(self):
        a = [[0, 2, 3], [0, 1, 0]]
        a = sp.coo_matrix(a)
        a = sparse_to_tensor(a)
        a = sparse_scalar_multiply(a, 2)
        a = tf.sparse_tensor_to_dense(a)
        expected = [[0, 4, 6], [0, 2, 0]]

        with self.test_session():
            self.assertAllEqual(a.eval(), expected)

    def test_sparse_subtract(self):
        a = [[0, 2, 3], [0, 1, 0]]
        a = sp.coo_matrix(a)
        a = sparse_to_tensor(a)

        b = [[1, 0, 2], [3, 1, 0]]
        b = sp.coo_matrix(b)
        b = sparse_to_tensor(b)

        c = sparse_subtract(a, b)
        c = tf.sparse_tensor_to_dense(c)
        expected = [[-1, 2, 1], [-3, 0, 0]]

        with self.test_session():
            self.assertAllEqual(c.eval(), expected)

    def test_sparse_tensor_diag_matmul(self):
        a = [[2, 3, 0], [1, 0, 2], [0, 3, 0]]
        a = sp.coo_matrix(a)
        a = sparse_to_tensor(a)

        diag = [2, 0.5, 3]
        diag = tf.constant(diag)

        b = sparse_tensor_diag_matmul(a, diag)
        b = tf.sparse_tensor_to_dense(b)
        expected = [[4, 6, 0], [0.5, 0, 1], [0, 9, 0]]

        with self.test_session():
            self.assertAllEqual(b.eval(), expected)

        b = sparse_tensor_diag_matmul(a, diag, transpose=True)
        b = tf.sparse_tensor_to_dense(b)
        expected = [[4, 1.5, 0], [2, 0, 6], [0, 1.5, 0]]

        with self.test_session():
            self.assertAllEqual(b.eval(), expected)
