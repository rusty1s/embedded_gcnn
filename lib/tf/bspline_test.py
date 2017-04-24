import tensorflow as tf
import numpy as np
from numpy import pi as PI
from numpy.testing import assert_almost_equal as assert_almost
import scipy.sparse as sp

from .bspline import base
from .convert import sparse_to_tensor


class BsplineTest(tf.test.TestCase):
    def test_base_K1_P2(self):
        adj_rad = [[
            0, 0.25 * PI, 0.5 * PI, 0.75 * PI, PI, 1.25 * PI, 1.5 * PI,
            1.75 * PI, 2 * PI
        ]]
        adj_rad = sp.coo_matrix(adj_rad, dtype=np.float32)
        adj_rad = sparse_to_tensor(adj_rad)

        with self.test_session():
            p0 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=2, p=0))
            self.assertAllEqual(p0.eval(), [[0, 1, 1, 1, 1, 0, 0, 0, 0]])
            p1 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=2, p=1))
            self.assertAllEqual(p1.eval(), [[0, 0, 0, 0, 0, 1, 1, 1, 1]])

    def test_base_K1_P4(self):
        adj_rad = [[
            0, 0.25 * PI, 0.5 * PI, 0.75 * PI, PI, 1.25 * PI, 1.5 * PI,
            1.75 * PI, 2 * PI
        ]]
        adj_rad = sp.coo_matrix(adj_rad, dtype=np.float32)
        adj_rad = sparse_to_tensor(adj_rad)

        with self.test_session():
            p0 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=4, p=0))
            self.assertAllEqual(p0.eval(), [[0, 1, 1, 0, 0, 0, 0, 0, 0]])
            p1 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=4, p=1))
            self.assertAllEqual(p1.eval(), [[0, 0, 0, 1, 1, 0, 0, 0, 0]])
            p2 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=4, p=2))
            self.assertAllEqual(p2.eval(), [[0, 0, 0, 0, 0, 1, 1, 0, 0]])
            p3 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=4, p=3))
            self.assertAllEqual(p3.eval(), [[0, 0, 0, 0, 0, 0, 0, 1, 1]])

    def test_base_K1_P8(self):
        adj_rad = [[
            0, 0.25 * PI, 0.5 * PI, 0.75 * PI, PI, 1.25 * PI, 1.5 * PI,
            1.75 * PI, 2 * PI
        ]]
        adj_rad = sp.coo_matrix(adj_rad, dtype=np.float32)
        adj_rad = sparse_to_tensor(adj_rad)

        with self.test_session():
            p0 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=8, p=0))
            self.assertAllEqual(p0.eval(), [[0, 1, 0, 0, 0, 0, 0, 0, 0]])
            p1 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=8, p=1))
            self.assertAllEqual(p1.eval(), [[0, 0, 1, 0, 0, 0, 0, 0, 0]])
            p2 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=8, p=2))
            self.assertAllEqual(p2.eval(), [[0, 0, 0, 1, 0, 0, 0, 0, 0]])
            p3 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=8, p=3))
            self.assertAllEqual(p3.eval(), [[0, 0, 0, 0, 1, 0, 0, 0, 0]])
            p4 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=8, p=4))
            self.assertAllEqual(p4.eval(), [[0, 0, 0, 0, 0, 1, 0, 0, 0]])
            p5 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=8, p=5))
            self.assertAllEqual(p5.eval(), [[0, 0, 0, 0, 0, 0, 1, 0, 0]])
            p6 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=8, p=6))
            self.assertAllEqual(p6.eval(), [[0, 0, 0, 0, 0, 0, 0, 1, 0]])
            p7 = tf.sparse_tensor_to_dense(base(adj_rad, K=1, P=8, p=7))
            self.assertAllEqual(p7.eval(), [[0, 0, 0, 0, 0, 0, 0, 0, 1]])

    def test_base_K2_P2(self):
        adj_rad = [[
            0, 0.25 * PI, 0.5 * PI, 0.75 * PI, PI, 1.25 * PI, 1.5 * PI,
            1.75 * PI, 2 * PI
        ]]
        adj_rad = sp.coo_matrix(adj_rad, dtype=np.float32)
        adj_rad = sparse_to_tensor(adj_rad)

        with self.test_session():
            p0 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=2, p=0))
            assert_almost(p0.eval(),
                          [[0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0]])
            p1 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=2, p=1))
            assert_almost(p1.eval(),
                          [[0, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1]])

    def test_base_K2_P4(self):
        adj_rad = [[
            0, 0.25 * PI, 0.5 * PI, 0.75 * PI, PI, 1.25 * PI, 1.5 * PI,
            1.75 * PI, 2 * PI
        ]]
        adj_rad = sp.coo_matrix(adj_rad, dtype=np.float32)
        adj_rad = sparse_to_tensor(adj_rad)

        with self.test_session():
            p0P4 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=4, p=0))
            self.assertAllEqual(p0P4.eval(), [[0, 0.5, 1, 0.5, 0, 0, 0, 0, 0]])
            p1P4 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=4, p=1))
            assert_almost(p1P4.eval(), [[0, 0, 0, 0.5, 1, 0.5, 0, 0, 0]], 6)
            p2P4 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=4, p=2))
            assert_almost(p2P4.eval(), [[0, 0, 0, 0, 0, 0.5, 1, 0.5, 0]], 6)
            p3P4 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=4, p=3))
            assert_almost(p3P4.eval(), [[0, 0.5, 0, 0, 0, 0, 0, 0.5, 1]], 6)

    def test_base_K2_P8(self):
        adj_rad = [[
            0, 0.25 * PI, 0.5 * PI, 0.75 * PI, PI, 1.25 * PI, 1.5 * PI,
            1.75 * PI, 2 * PI
        ]]
        adj_rad = sp.coo_matrix(adj_rad, dtype=np.float32)
        adj_rad = sparse_to_tensor(adj_rad)

        with self.test_session():
            p0 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=8, p=0))
            self.assertAllEqual(p0.eval(), [[0, 1, 0, 0, 0, 0, 0, 0, 0]])
            p1 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=8, p=1))
            self.assertAllEqual(p1.eval(), [[0, 0, 1, 0, 0, 0, 0, 0, 0]])
            p2 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=8, p=2))
            self.assertAllEqual(p2.eval(), [[0, 0, 0, 1, 0, 0, 0, 0, 0]])
            p3 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=8, p=3))
            assert_almost(p3.eval(), [[0, 0, 0, 0, 1, 0, 0, 0, 0]], 6)
            p4 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=8, p=4))
            assert_almost(p4.eval(), [[0, 0, 0, 0, 0, 1, 0, 0, 0]], 6)
            p5 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=8, p=5))
            assert_almost(p5.eval(), [[0, 0, 0, 0, 0, 0, 1, 0, 0]], 6)
            p6 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=8, p=6))
            assert_almost(p6.eval(), [[0, 0, 0, 0, 0, 0, 0, 1, 0]], 6)
            p7 = tf.sparse_tensor_to_dense(base(adj_rad, K=2, P=8, p=7))
            assert_almost(p7.eval(), [[0, 0, 0, 0, 0, 0, 0, 0, 1]], 6)
