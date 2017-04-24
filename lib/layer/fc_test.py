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
        inputs = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertAllEqual(
                layer(inputs).eval(),
                tf.nn.relu(
                    tf.nn.bias_add(
                        tf.matmul(inputs, layer.vars['weights']), layer.vars[
                            'bias'])).eval())

    def test_call_with_dropout(self):
        layer = FC(3, 5, dropout=0.5, name='call_with_dropout')
        inputs = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            layer(inputs).eval()  # Dropout is random and not testable.
