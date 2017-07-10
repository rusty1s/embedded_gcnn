import numpy as np
from numpy.testing import assert_equal
import scipy.sparse as sp
import tensorflow as tf

from .math import (sparse_scalar_multiply, sparse_tensor_diag_matmul,
                   _diag_matmul_py, _diag_matmul_transpose_py)
from .convert import sparse_to_tensor


class MathTest(tf.test.TestCase):
    def test_sparse_scalar_multiply(self):
        a = [[0, 2, 3], [0, 1, 0]]
        a = sp.coo_matrix(a)
        a = sparse_to_tensor(a)
        a = sparse_scalar_multiply(a, 2)
        a = tf.sparse_tensor_to_dense(a)
        expected = [[0, 4, 6], [0, 2, 0]]

        with self.test_session():
            self.assertAllEqual(a.eval(), expected)

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

    def test_diag_matmul_py(self):
        indices = np.array([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1]])
        values = np.array([2, 3, 1, 2, 3])
        diag = np.array([2, 0.5, 3])

        result = _diag_matmul_py(indices, values, diag)
        expected = [4, 6, 0.5, 1, 9]
        assert_equal(result, expected)

    def test_diag_matmul_transpose_py(self):
        indices = np.array([[1, 0], [0, 0], [0, 1], [1, 2], [2, 1]])
        values = np.array([1, 2, 3, 2, 3])
        diag = np.array([2, 0.5, 3])

        result = _diag_matmul_transpose_py(indices, values, diag)
        expected = [2, 4, 1.5, 6, 1.5]
        assert_equal(result, expected)
