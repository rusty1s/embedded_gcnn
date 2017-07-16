import tensorflow as tf

from .conv2d import conv, Conv2d


class Conv2dTest(tf.test.TestCase):
    def test_conv(self):
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        inputs = tf.constant(image, tf.float32)
        inputs = tf.reshape(inputs, [1, 3, 3, 1])

        weights = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        weights = tf.constant(weights, tf.float32)
        weights = tf.reshape(weights, [3, 3, 1, 1])

        outputs = conv(inputs, weights, stride=1)

        expected = [[
            [
                [1 * 5 + 2 * 6 + 4 * 8 + 5 * 9],
                [1 * 4 + 2 * 5 + 3 * 6 + 4 * 7 + 5 * 8 + 6 * 9],
                [2 * 4 + 3 * 5 + 5 * 7 + 6 * 8],
            ],
            [
                [1 * 2 + 2 * 3 + 4 * 5 + 5 * 6 + 7 * 8 + 8 * 9],
                [1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81],
                [2 * 1 + 3 * 2 + 5 * 4 + 6 * 5 + 8 * 7 + 9 * 8],
            ],
            [
                [4 * 2 + 5 * 3 + 7 * 5 + 8 * 6],
                [4 * 1 + 5 * 2 + 6 * 3 + 7 * 4 + 5 * 8 + 9 * 6],
                [5 * 1 + 6 * 2 + 8 * 4 + 9 * 5],
            ],
        ]]

        with self.test_session():
            self.assertEqual(outputs.eval().shape, (1, 3, 3, 1))
            self.assertAllEqual(outputs.eval(), expected)

    def test_init(self):
        layer = Conv2d(1, 2)
        self.assertEqual(layer.name, 'conv2d_1')
        self.assertIsNone(layer.dropout)
        self.assertEqual(layer.stride, 1)
        self.assertEqual(layer.vars['weights'].get_shape(), [3, 3, 1, 2])
        self.assertEqual(layer.vars['bias'].get_shape(), [2])

        layer = Conv2d(2, 3, size=5, stride=4, dropout=0.5)
        self.assertEqual(layer.name, 'conv2d_2')
        self.assertEqual(layer.dropout, 0.5)
        self.assertEqual(layer.stride, 4)
        self.assertEqual(layer.vars['weights'].get_shape(), [5, 5, 2, 3])
        self.assertEqual(layer.vars['bias'].get_shape(), [3])

    def test_call(self):
        layer = Conv2d(1, 2, name='call')

        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        inputs = tf.constant(image, tf.float32)
        inputs = tf.reshape(inputs, [1, 3, 3, 1])

        outputs = layer(inputs)

        expected = conv(inputs, layer.vars['weights'], stride=1)
        expected = tf.nn.bias_add(expected, layer.vars['bias'])
        expected = tf.nn.relu(expected)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(outputs.eval().shape, (1, 3, 3, 2))
            self.assertAllEqual(outputs.eval(), expected.eval())

    def test_call_without_bias(self):
        layer = Conv2d(1, 2, bias=False, name='call_wihtout_bias')

        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        inputs = tf.constant(image, tf.float32)
        inputs = tf.reshape(inputs, [1, 3, 3, 1])

        outputs = layer(inputs)

        expected = conv(inputs, layer.vars['weights'], stride=1)
        expected = tf.nn.relu(expected)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(outputs.eval().shape, (1, 3, 3, 2))
            self.assertAllEqual(outputs.eval(), expected.eval())

    def test_call_with_dropout(self):
        layer = Conv2d(1, 2, dropout=0.5, name='call_with_dropout')

        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        inputs = tf.constant(image, tf.float32)
        inputs = tf.reshape(inputs, [1, 3, 3, 1])

        outputs = layer(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Dropout is random and therefore not testable, so we just ran it
            # and ensure that the computation succeeds.
            self.assertEqual(outputs.eval().shape, (1, 3, 3, 2))
