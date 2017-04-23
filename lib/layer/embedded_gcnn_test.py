import tensorflow as tf
import numpy as np
from numpy import pi as PI
from numpy.testing import assert_almost_equal as assert_almost
import scipy.sparse as sp

from .embedded_gcnn import base, conv, EmbeddedGCNN
from ..graph.sparse import sparse_to_tensor


class EmbeddedGCNNTest(tf.test.TestCase):
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

    def test_conv_K2_P4(self):
        features = [[1, 2], [3, 4], [5, 6], [7, 8]]
        features = tf.constant(features, dtype=tf.float32)

        adj_dist = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj_dist = sp.coo_matrix(adj_dist, dtype=np.float32)
        adj_dist = sparse_to_tensor(adj_dist)

        adj_rad = [[0, 0.25 * PI, 0.75 * PI, 0], [1.25 * PI, 0, 0, 0.25 * PI],
                   [1.75 * PI, 0, 0, 0.25 * PI], [0, 1.75 * PI, 1.25 * PI, 0]]
        adj_rad = sp.coo_matrix(adj_rad, dtype=np.float32)
        adj_rad = sparse_to_tensor(adj_rad)

        # Shape of weights:
        # P = 4, 2 input features, 1 output feature => [5, 2, 1].
        weights = [[[0.1], [0.9]], [[0.7], [0.3]], [[0.4], [0.6]],
                   [[0.8], [0.2]], [[0.5], [0.5]]]
        weights = tf.constant(weights, dtype=tf.float32)

        output = conv(features, adj_dist, adj_rad, weights, K=2)

        expected_1 = 1 * 0.5 + 2 * 0.5
        expected_1 += 2 * 0.5 * (3 * 0.1 + 4 * 0.9)
        expected_1 += 2 * 0.5 * (3 * 0.8 + 4 * 0.2)
        expected_1 += 1 * 0.5 * (5 * 0.7 + 6 * 0.3)
        expected_1 += 1 * 0.5 * (5 * 0.1 + 6 * 0.9)

        expected_2 = 3 * 0.5 + 4 * 0.5
        expected_2 += 2 * 0.5 * (1 * 0.4 + 2 * 0.6)
        expected_2 += 2 * 0.5 * (1 * 0.7 + 2 * 0.3)
        expected_2 += 1 * 0.5 * (7 * 0.8 + 8 * 0.2)
        expected_2 += 1 * 0.5 * (7 * 0.1 + 8 * 0.9)

        expected_3 = 5 * 0.5 + 6 * 0.5
        expected_3 += 1 * 0.5 * (1 * 0.8 + 2 * 0.2)
        expected_3 += 1 * 0.5 * (1 * 0.4 + 2 * 0.6)
        expected_3 += 2 * 0.5 * (7 * 0.1 + 8 * 0.9)
        expected_3 += 2 * 0.5 * (7 * 0.8 + 8 * 0.2)

        expected_4 = 7 * 0.5 + 8 * 0.5
        expected_4 += 1 * 0.5 * (3 * 0.8 + 4 * 0.2)
        expected_4 += 1 * 0.5 * (3 * 0.4 + 4 * 0.6)
        expected_4 += 2 * 0.5 * (5 * 0.7 + 6 * 0.3)
        expected_4 += 2 * 0.5 * (5 * 0.4 + 6 * 0.6)

        expected = [[expected_1], [expected_2], [expected_3], [expected_4]]

        with self.test_session():
            assert_almost(output.eval(), expected, 6)

    def test_init(self):
        layer = EmbeddedGCNN(
            1,
            2,
            adjs_dist=None,
            adjs_rad=None,
            local_controllability=2,
            sampling_points=8)
        self.assertEqual(layer.name, 'embeddedgcnn_1')
        self.assertEqual(layer.adjs_dist, None)
        self.assertEqual(layer.adjs_rad, None)
        self.assertEqual(layer.K, 2)
        self.assertEqual(layer.P, 8)
        self.assertIn('weights', layer.vars)
        self.assertEqual(layer.vars['weights'].get_shape(), [9, 1, 2])
        self.assertIn('bias', layer.vars)
        self.assertEqual(layer.vars['bias'].get_shape(), [2])
