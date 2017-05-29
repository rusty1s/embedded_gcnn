import tensorflow as tf
from numpy.testing import assert_almost_equal

from .inits import weight_variable, bias_variable


class InitsTest(tf.test.TestCase):
    def test_weight_variable(self):
        weights = weight_variable([2, 3], name='weights_1')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(weights.name, 'weights_1:0')
            assert_almost_equal(weights.eval().shape, (2, 3))

        weights = weight_variable(
            [2, 3], name='weights_2', stddev=0, dtype=tf.float64)

        expected = [[0, 0, 0], [0, 0, 0]]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(weights.name, 'weights_2:0')
            self.assertAllEqual(weights.eval(), expected)
            self.assertEqual(tf.get_collection('losses'), [])

    def test_weight_variable_with_decay(self):
        weights = weight_variable([2, 3], name='weights', decay=0.01)
        losses = tf.get_collection('losses')

        expected = tf.nn.l2_loss(weights)
        expected = expected * 0.01

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(len(losses), 1)
            self.assertEqual(losses[0].name, 'weight_loss:0')
            self.assertEqual(losses[0].eval(), expected.eval())

    def test_bias_variable(self):
        bias = bias_variable([2, 3], 'bias_1')

        expected = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(bias.name, 'bias_1:0')
            assert_almost_equal(bias.eval(), expected)

        bias = bias_variable([1, 4], name='bias_2', constant=1, dtype=tf.uint8)

        expected = [[1, 1, 1, 1]]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(bias.name, 'bias_2:0')
            self.assertAllEqual(bias.eval(), expected)

    def test_bias_variable_with_decay(self):
        bias = bias_variable([2, 3], name='biases', decay=0.01)
        losses = tf.get_collection('losses')

        expected = tf.nn.l2_loss(bias)
        expected = expected * 0.01

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(len(losses), 1)
            self.assertEqual(losses[0].name, 'bias_loss:0')
            self.assertEqual(losses[0].eval(), expected.eval())
