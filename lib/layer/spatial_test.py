import tensorflow as tf

from .spatial import conv, SpatialCNN


class SpatialCNNTest(tf.test.TestCase):
    def test_conv(self):
        cube = [
            [[1, 2], [3, 4], [5, 6]],
            [[6, 5], [4, 3], [2, 1]],
        ]

        inputs = tf.constant([cube, cube], tf.float32)

        weights = tf.constant(1, tf.float32, shape=[1, 3, 2, 8])
        self.assertAllEqual(conv(inputs, weights).get_shape(), [2, 2, 8])

        weights = [1, 2, 3, 4, 5, 6]
        weights = tf.constant(weights, tf.float32)
        weights = tf.reshape(weights, [1, 3, 2, 1])

        outputs = conv(inputs, weights)

        expected = [[1 * 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6],
                    [1 * 6 + 2 * 5 + 3 * 4 + 4 * 3 + 5 * 2 + 6 * 1]]

        with self.test_session():
            self.assertAllEqual(outputs.eval()[0], expected)
            self.assertAllEqual(outputs.eval()[1], expected)

    def test_init(self):
        layer = SpatialCNN(3, 64, 9)
        self.assertEqual(layer.name, 'spatialcnn_1')
        self.assertEqual(layer.vars['weights'].get_shape(), [1, 9, 3, 64])
        self.assertEqual(layer.vars['bias'].get_shape(), [64])

    def test_call(self):
        layer = SpatialCNN(3, 64, 9, name='call')

        inputs = tf.constant(1, tf.float32, shape=[4, 25, 9, 3])

        outputs = layer(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(outputs.eval().shape, (4, 25, 64))
