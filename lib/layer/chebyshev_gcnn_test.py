import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from .chebyshev_gcnn import ChebyshevGCNN
from ..graph.sparse import sparse_to_tensor


class ChebyshevGCNNTest(tf.test.TestCase):
    def test_init(self):
        layer = ChebyshevGCNN(1, 2, laps=None, degree=3)
        self.assertEqual(layer.name, 'chebyshevgcnn_1')
        self.assertEqual(layer.laps, None)
        self.assertIn('weights', layer.vars)
        self.assertEqual(layer.vars['weights'].get_shape(), [4, 1, 2])
        self.assertIn('bias', layer.vars)
        self.assertEqual(layer.vars['bias'].get_shape(), [2])

        layer = ChebyshevGCNN(3, 4, laps=None, degree=5, bias=False)
        self.assertEqual(layer.name, 'chebyshevgcnn_2')
        self.assertEqual(layer.laps, None)
        self.assertIn('weights', layer.vars)
        self.assertEqual(layer.vars['weights'].get_shape(), [6, 3, 4])
        self.assertNotIn('bias', layer.vars)

    def test_call(self):
        lap = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        lap = sp.coo_matrix(lap, dtype=np.float32)
        lap = sparse_to_tensor(lap)

        layer = ChebyshevGCNN(
            2, 3, laps=[lap, lap], degree=3, name='call_single')
        input_1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        input_2 = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]

        inputs = tf.constant([input_1, input_2])

        outputs = layer(inputs)

        Tx_1_0 = inputs[0]
        output_1 = tf.matmul(Tx_1_0, layer.vars['weights'][0])
        Tx_1_1 = tf.sparse_tensor_dense_matmul(lap, inputs[0])
        output_1 = tf.add(
            tf.matmul(Tx_1_1, layer.vars['weights'][1]), output_1)
        Tx_1_2 = 2 * tf.sparse_tensor_dense_matmul(lap, Tx_1_1) - Tx_1_0
        output_1 = tf.add(
            tf.matmul(Tx_1_2, layer.vars['weights'][2]), output_1)
        Tx_1_3 = 2 * tf.sparse_tensor_dense_matmul(lap, Tx_1_2) - Tx_1_1
        output_1 = tf.add(
            tf.matmul(Tx_1_3, layer.vars['weights'][3]), output_1)
        output_1 = tf.nn.bias_add(output_1, layer.vars['bias'])
        output_1 = tf.nn.relu(output_1)

        Tx_2_0 = inputs[1]
        output_2 = tf.matmul(Tx_2_0, layer.vars['weights'][0])
        Tx_2_1 = tf.sparse_tensor_dense_matmul(lap, inputs[1])
        output_2 = tf.add(
            tf.matmul(Tx_2_1, layer.vars['weights'][1]), output_2)
        Tx_2_2 = 2 * tf.sparse_tensor_dense_matmul(lap, Tx_2_1) - Tx_2_0
        output_2 = tf.add(
            tf.matmul(Tx_2_2, layer.vars['weights'][2]), output_2)
        Tx_2_3 = 2 * tf.sparse_tensor_dense_matmul(lap, Tx_2_2) - Tx_2_1
        output_2 = tf.add(
            tf.matmul(Tx_2_3, layer.vars['weights'][3]), output_2)
        output_2 = tf.nn.bias_add(output_2, layer.vars['bias'])
        output_2 = tf.nn.relu(output_2)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertAllEqual(outputs.eval().shape, [2, 3, 3])
            self.assertAllEqual(outputs[0].eval(), output_1.eval())
            self.assertAllEqual(outputs[1].eval(), output_2.eval())
