import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from .gcnn import conv, GCNN
from ..tf.convert import sparse_to_tensor


class GCNNTest(tf.test.TestCase):
    def test_conv(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj_norm = adj + sp.eye(3, dtype=np.float32)
        degree = np.array(adj_norm.sum(1)).flatten()
        degree = np.power(degree, -0.5)
        degree = sp.diags(degree)
        adj_norm = degree.dot(adj_norm).dot(degree)
        adj = sparse_to_tensor(adj)

        features = [[1, 2], [3, 4], [5, 6]]
        features_np = np.array(features, dtype=np.float32)
        features_tf = tf.constant(features, dtype=tf.float32)

        weights = [[0.3], [0.7]]
        weights_np = np.array(weights, dtype=np.float32)
        weights_tf = tf.constant(weights, dtype=tf.float32)

        expected = adj_norm.dot(features_np).dot(weights_np)

        with self.test_session():
            self.assertAllEqual(
                conv(features_tf, adj, weights_tf).eval(), expected)

    def test_init(self):
        layer = GCNN(1, 2, adjs=None)
        self.assertEqual(layer.name, 'gcnn_1')
        self.assertIsNone(layer.adjs)
        self.assertEqual(layer.vars['weights'].get_shape(), [1, 2])
        self.assertEqual(layer.vars['bias'].get_shape(), [2])

    def test_call(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = sp.coo_matrix(adj)
        adj = sparse_to_tensor(adj)

        layer = GCNN(2, 3, [adj, adj], name='call')

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
        adj = sp.coo_matrix(adj)
        adj = sparse_to_tensor(adj)

        layer = GCNN(2, 3, [adj, adj], bias=False, name='call_without_bias')

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
