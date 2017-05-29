import tensorflow as tf
import numpy as np
from numpy import pi as PI
from numpy.testing import assert_almost_equal
import scipy.sparse as sp

from .embedded_gcnn import conv, EmbeddedGCNN
from ..tf.convert import sparse_to_tensor


class EmbeddedGCNNTest(tf.test.TestCase):
    def test_conv_K2_P4(self):
        features = [[1, 2], [3, 4], [5, 6], [7, 8]]
        features = tf.constant(features, dtype=tf.float32)

        adj_dist = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj_dist = sp.coo_matrix(adj_dist, dtype=np.float32)
        adj_dist = sparse_to_tensor(adj_dist)

        adj_rad = [[0, 0.25 * PI, 0.75 * PI, 0], [1.25 * PI, 0, 0, 0.75 * PI],
                   [1.75 * PI, 0, 0, 0.25 * PI], [0, 1.75 * PI, 1.25 * PI, 0]]
        adj_rad = sp.coo_matrix(adj_rad, dtype=np.float32)
        adj_rad = sparse_to_tensor(adj_rad)

        weights = [[[0.1], [0.9]], [[0.7], [0.3]], [[0.4], [0.6]],
                   [[0.8], [0.2]], [[0.5], [0.5]]]
        weights = tf.constant(weights, dtype=tf.float32)

        expected_1 = 1 * 0.5 * 0.25 + 2 * 0.5 * 0.25
        expected_1 += 2 * 0.5 * 0.25 * (3 * 0.1 + 4 * 0.9)
        expected_1 += 2 * 0.5 * 0.25 * (3 * 0.8 + 4 * 0.2)
        expected_1 += 1 * 0.5 * 0.25 * (5 * 0.7 + 6 * 0.3)
        expected_1 += 1 * 0.5 * 0.25 * (5 * 0.1 + 6 * 0.9)

        expected_2 = 3 * 0.5 * 0.25 + 4 * 0.5 * 0.25
        expected_2 += 2 * 0.5 * 0.25 * (1 * 0.4 + 2 * 0.6)
        expected_2 += 2 * 0.5 * 0.25 * (1 * 0.7 + 2 * 0.3)
        expected_2 += 1 * 0.5 * 0.25 * (7 * 0.7 + 8 * 0.3)
        expected_2 += 1 * 0.5 * 0.25 * (7 * 0.1 + 8 * 0.9)

        expected_3 = 5 * 0.5 * 0.25 + 6 * 0.5 * 0.25
        expected_3 += 1 * 0.5 * 0.25 * (1 * 0.8 + 2 * 0.2)
        expected_3 += 1 * 0.5 * 0.25 * (1 * 0.4 + 2 * 0.6)
        expected_3 += 2 * 0.5 * 0.25 * (7 * 0.1 + 8 * 0.9)
        expected_3 += 2 * 0.5 * 0.25 * (7 * 0.8 + 8 * 0.2)

        expected_4 = 7 * 0.5 * 0.25 + 8 * 0.5 * 0.25
        expected_4 += 1 * 0.5 * 0.25 * (3 * 0.8 + 4 * 0.2)
        expected_4 += 1 * 0.5 * 0.25 * (3 * 0.4 + 4 * 0.6)
        expected_4 += 2 * 0.5 * 0.25 * (5 * 0.7 + 6 * 0.3)
        expected_4 += 2 * 0.5 * 0.25 * (5 * 0.4 + 6 * 0.6)

        expected = [[expected_1], [expected_2], [expected_3], [expected_4]]

        with self.test_session():
            assert_almost_equal(
                conv(features, adj_dist, adj_rad, weights, K=2).eval(),
                expected, 6)

    def test_init(self):
        layer = EmbeddedGCNN(
            1,
            2,
            adjs_dist=None,
            adjs_rad=None,
            local_controllability=2,
            sampling_points=8)
        self.assertEqual(layer.name, 'embeddedgcnn_1')
        self.assertIsNone(layer.adjs_dist)
        self.assertIsNone(layer.adjs_rad)
        self.assertEqual(layer.K, 2)
        self.assertEqual(layer.P, 8)
        self.assertEqual(layer.vars['weights'].get_shape(), [9, 1, 2])
        self.assertEqual(layer.vars['bias'].get_shape(), [2])

    def test_call(self):
        adj_dist = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj_dist = sp.coo_matrix(adj_dist, dtype=np.float32)
        adj_dist = sparse_to_tensor(adj_dist)

        adj_rad = [[0, 0.25 * PI, 0.75 * PI, 0], [1.25 * PI, 0, 0, 0.25 * PI],
                   [1.75 * PI, 0, 0, 0.25 * PI], [0, 1.75 * PI, 1.25 * PI, 0]]
        adj_rad = sp.coo_matrix(adj_rad, dtype=np.float32)
        adj_rad = sparse_to_tensor(adj_rad)

        layer = EmbeddedGCNN(
            2,
            3, [adj_dist, adj_dist], [adj_rad, adj_rad],
            local_controllability=1,
            sampling_points=4,
            name='call')

        input_1 = [[1, 2], [3, 4], [5, 6], [7, 8]]
        input_1 = tf.constant(input_1, dtype=tf.float32)
        input_2 = [[9, 10], [11, 12], [13, 14], [15, 16]]
        input_2 = tf.constant(input_2, dtype=tf.float32)
        inputs = [input_1, input_2]
        outputs = layer(inputs)

        expected_1 = conv(
            input_1, adj_dist, adj_rad, layer.vars['weights'], K=1)
        expected_1 = tf.nn.bias_add(expected_1, layer.vars['bias'])
        expected_1 = tf.nn.relu(expected_1)

        expected_2 = conv(
            input_2, adj_dist, adj_rad, layer.vars['weights'], K=1)
        expected_2 = tf.nn.bias_add(expected_2, layer.vars['bias'])
        expected_2 = tf.nn.relu(expected_2)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(len(outputs), 2)
            self.assertEqual(outputs[0].eval().shape, (4, 3))
            self.assertEqual(outputs[1].eval().shape, (4, 3))
            self.assertAllEqual(outputs[0].eval(), expected_1.eval())
            self.assertAllEqual(outputs[1].eval(), expected_2.eval())

    def test_call_without_bias(self):
        adj_dist = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj_dist = sp.coo_matrix(adj_dist, dtype=np.float32)
        adj_dist = sparse_to_tensor(adj_dist)

        adj_rad = [[0, 0.25 * PI, 0.75 * PI, 0], [1.25 * PI, 0, 0, 0.25 * PI],
                   [1.75 * PI, 0, 0, 0.25 * PI], [0, 1.75 * PI, 1.25 * PI, 0]]
        adj_rad = sp.coo_matrix(adj_rad, dtype=np.float32)
        adj_rad = sparse_to_tensor(adj_rad)

        layer = EmbeddedGCNN(
            2,
            3, [adj_dist, adj_dist], [adj_rad, adj_rad],
            local_controllability=1,
            sampling_points=4,
            bias=False,
            name='call_without_bias')

        input_1 = [[1, 2], [3, 4], [5, 6], [7, 8]]
        input_1 = tf.constant(input_1, dtype=tf.float32)
        input_2 = [[9, 10], [11, 12], [13, 14], [15, 16]]
        input_2 = tf.constant(input_2, dtype=tf.float32)
        inputs = [input_1, input_2]
        outputs = layer(inputs)

        expected_1 = conv(
            input_1, adj_dist, adj_rad, layer.vars['weights'], K=1)
        expected_1 = tf.nn.relu(expected_1)

        expected_2 = conv(
            input_2, adj_dist, adj_rad, layer.vars['weights'], K=1)
        expected_2 = tf.nn.relu(expected_2)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(len(outputs), 2)
            self.assertEqual(outputs[0].eval().shape, (4, 3))
            self.assertEqual(outputs[1].eval().shape, (4, 3))
            self.assertAllEqual(outputs[0].eval(), expected_1.eval())
            self.assertAllEqual(outputs[1].eval(), expected_2.eval())
