import numpy as np
import tensorflow as tf

from .gcnn import GCNN


class GCNNTest(tf.test.TestCase):
    def test_init(self):
        layer = GCNN(1, 2)
        self.assertEqual(layer.name, 'gcnn_1')
        self.assertEqual(layer.act, tf.nn.relu)
        self.assertEqual(layer.bias, True)
        self.assertEqual(layer.logging, False)
        self.assertIn('weights', layer.vars)
        self.assertEqual(layer.vars['weights'].get_shape(), [1, 2])
        self.assertIn('bias', layer.vars)
        self.assertEqual(layer.vars['bias'].get_shape(), [2])

        layer = GCNN(3, 4, bias=False, logging=True)
        self.assertEqual(layer.name, 'gcnn_2')
        self.assertEqual(layer.act, tf.nn.relu)
        self.assertEqual(layer.bias, False)
        self.assertEqual(layer.logging, True)
        self.assertIn('weights', layer.vars)
        self.assertEqual(layer.vars['weights'].get_shape(), [3, 4])
        self.assertNotIn('bias', layer.vars)

    def test_bias_constant(self):
        layer1 = GCNN(2, 3, name='bias_1')
        layer2 = GCNN(2, 3, bias_constant=1.0, name='bias_2')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertAllEqual(layer1.vars['bias'].eval(),
                                np.array([0.1, 0.1, 0.1], dtype=np.float32))
            self.assertAllEqual(layer2.vars['bias'].eval(),
                                np.array([1.0, 1.0, 1.0], dtype=np.float32))

    def test_call(self):
        pass
        # layer = GCNN(1, 3, bias_constant=1, name='call')
        # adj = tf.SparseTensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 1, 1]],
        #                       [1.0, 1.0, 1.0, 1.0], [2, 3, 3])
        # inputs = tf.constant([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
        # outputs = layer(inputs, adj=adj)

        # with self.test_session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     outputs = outputs.eval()

        #     self.assertAllEqual(
        #         outputs[0],
        #         tf.nn.relu(tf.matmul(inputs[0], layer.vars['weights']) +
        #                    1).eval())

        #     self.assertAllEqual(
        #         outputs[1],
        #         tf.nn.relu(
        #             tf.matmul(
        #                 tf.matmul(
        #                     tf.sparse_tensor_to_dense(adj)[1], inputs[1]),
        #                 layer.vars['weights']) + 1).eval())
