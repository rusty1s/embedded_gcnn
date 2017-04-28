import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from .chebyshev_gcnn import conv, ChebyshevGCNN
from ..tf.convert import sparse_to_tensor
from ..tf.laplacian import laplacian, rescale_lap


class ChebyshevGCNNTest(tf.test.TestCase):
    def test_conv_K2(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj = sparse_to_tensor(adj)
        lap = laplacian(adj)
        lap = rescale_lap(lap)

        features = [[1, 2], [3, 4], [5, 6]]
        features = tf.constant(features, dtype=tf.float32)

        weights = [[[0.3], [0.7]], [[0.4], [0.6]], [[0.8], [0.2]]]
        weights = tf.constant(weights, dtype=tf.float32)

        Tx_0 = features
        expected = tf.matmul(Tx_0, weights[0])
        Tx_1 = tf.sparse_tensor_dense_matmul(lap, features)
        expected = tf.add(tf.matmul(Tx_1, weights[1]), expected)
        Tx_2 = 2 * tf.sparse_tensor_dense_matmul(lap, Tx_1) - Tx_0
        expected = tf.add(tf.matmul(Tx_2, weights[2]), expected)

        with self.test_session():
            self.assertAllEqual(
                conv(features, adj, weights).eval(), expected.eval())

    def test_init(self):
        layer = ChebyshevGCNN(1, 2, adjs=None, degree=3)
        self.assertEqual(layer.name, 'chebyshevgcnn_1')
        self.assertIsNone(layer.adjs)
        self.assertEqual(layer.vars['weights'].get_shape(), [4, 1, 2])
        self.assertEqual(layer.vars['bias'].get_shape(), [2])

    def test_call(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj = sparse_to_tensor(adj)

        layer = ChebyshevGCNN(2, 3, [adj, adj], degree=3, name='call')

        input_1 = [[1, 2], [3, 4], [5, 6]]
        input_1 = tf.constant(input_1, dtype=tf.float32)
        input_2 = [[7, 8], [9, 10], [11, 12]]
        input_2 = tf.constant(input_2, dtype=tf.float32)
        inputs = [input_1, input_2]
        outputs = layer(inputs)

        expected_1 = conv(input_1, adj, layer.vars['weights'])
        expected_1 = tf.nn.bias_add(expected_1, layer.vars['bias'])
        expected_1 = tf.nn.relu(expected_1)

        expected_2 = conv(input_2, adj, layer.vars['weights'])
        expected_2 = tf.nn.bias_add(expected_2, layer.vars['bias'])
        expected_2 = tf.nn.relu(expected_2)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(len(outputs), 2)
            self.assertEqual(outputs[0].eval().shape, (3, 3))
            self.assertEqual(outputs[1].eval().shape, (3, 3))
            self.assertAllEqual(outputs[0].eval(), expected_1.eval())

    def test_call_without_bias(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj = sparse_to_tensor(adj)

        layer = ChebyshevGCNN(
            2, 3, [adj, adj], degree=3, bias=False, name='call_without_bias')

        input_1 = [[1, 2], [3, 4], [5, 6]]
        input_1 = tf.constant(input_1, dtype=tf.float32)
        input_2 = [[7, 8], [9, 10], [11, 12]]
        input_2 = tf.constant(input_2, dtype=tf.float32)
        inputs = [input_1, input_2]
        outputs = layer(inputs)

        expected_1 = conv(input_1, adj, layer.vars['weights'])
        expected_1 = tf.nn.relu(expected_1)

        expected_2 = conv(input_2, adj, layer.vars['weights'])
        expected_2 = tf.nn.relu(expected_2)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(len(outputs), 2)
            self.assertEqual(outputs[0].eval().shape, (3, 3))
            self.assertEqual(outputs[1].eval().shape, (3, 3))
            self.assertAllEqual(outputs[0].eval(), expected_1.eval())
            self.assertAllEqual(outputs[1].eval(), expected_2.eval())
            self.assertAllEqual(outputs[1].eval(), expected_2.eval())
