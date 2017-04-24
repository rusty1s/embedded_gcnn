# import numpy as np
import tensorflow as tf

from .gcnn import GCNN


class GCNNTest(tf.test.TestCase):
    def test_conv(self):
        pass

    def test_init(self):
        layer = GCNN(1, 2, adjs=None)
        self.assertEqual(layer.name, 'gcnn_1')
        self.assertIsNone(layer.adjs)
        self.assertEqual(layer.vars['weights'].get_shape(), [1, 2])
        self.assertEqual(layer.vars['bias'].get_shape(), [2])

    def test_call(self):
        pass
        # adj = tf.SparseTensor([[0, 1], [1, 0], [1, 2], [2, 1]],
        #                       [1.0, 1.0, 2.0, 2.0], [3, 3])

        # layer = GCNN(2, 3, adj, name='call_single')
        # input_1 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        # input_2 = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]

        # inputs = tf.constant([input_1, input_2])  # batch_size = 2

        # outputs = layer(inputs)

        # output_1 = tf.sparse_tensor_dense_matmul(adj, tf.constant(input_1))
        # output_1 = tf.matmul(output_1, layer.vars['weights'])
        # output_1 = tf.nn.bias_add(output_1, layer.vars['bias'])
        # output_1 = tf.nn.relu(output_1)

        # output_2 = tf.sparse_tensor_dense_matmul(adj, tf.constant(input_2))
        # output_2 = tf.matmul(output_2, layer.vars['weights'])
        # output_2 = tf.nn.bias_add(output_2, layer.vars['bias'])
        # output_2 = tf.nn.relu(output_2)

        # with self.test_session() as sess:
        #     sess.run(tf.global_variables_initializer())

        #     self.assertAllEqual(outputs.eval().shape, [2, 3, 3])
        #     self.assertAllEqual(outputs[0].eval(), output_1.eval())
        #     self.assertAllEqual(outputs[1].eval(), output_2.eval())
