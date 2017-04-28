import tensorflow as tf

from .fc import FC


class FCTest(tf.test.TestCase):
    def test_init(self):
        layer = FC(1, 2)
        self.assertEqual(layer.name, 'fc_1')
        self.assertIsNone(layer.dropout)
        self.assertIn('weights', layer.vars)
        self.assertEqual(layer.vars['weights'].get_shape(), [1, 2])
        self.assertEqual(layer.vars['bias'].get_shape(), [2])

        layer = FC(1, 2, dropout=0.5)
        self.assertEqual(layer.dropout, 0.5)

    def test_call(self):
        layer = FC(3, 5, name='call')
        inputs = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
        outputs = layer(inputs)

        expected = tf.matmul(inputs, layer.vars['weights'])
        expected = tf.nn.bias_add(expected, layer.vars['bias'])
        expected = tf.nn.relu(expected)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertAllEqual(outputs.eval().shape, (2, 5))
            self.assertAllEqual(outputs.eval(), expected.eval())

    def test_call_without_bias(self):
        layer = FC(3, 5, bias=False, name='call_without_bias')
        inputs = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
        outputs = layer(inputs)

        expected = tf.matmul(inputs, layer.vars['weights'])
        expected = tf.nn.relu(expected)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertAllEqual(outputs.eval().shape, (2, 5))
            self.assertAllEqual(outputs.eval(), expected.eval())

    def test_call_with_dropout(self):
        layer = FC(3, 5, dropout=0.5, name='call_with_dropout')
        inputs = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
        outputs = layer(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Dropout is random and therefore not testable, so we just ran it
            # and ensure that the computation succeeds.
            self.assertEqual(outputs.eval().shape, (2, 5))
