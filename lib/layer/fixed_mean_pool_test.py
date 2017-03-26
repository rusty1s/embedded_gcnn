import tensorflow as tf

from .fixed_mean_pool import FixedMeanPool


class FixedMeanPoolTest(tf.test.TestCase):
    def test_init(self):
        layer = FixedMeanPool()
        self.assertEqual(layer.name, 'fixedmeanpool_1')
        self.assertEqual(layer.logging, False)

    def test_call(self):
        layer = FixedMeanPool(name='call')
        input_1 = [[1, 2], [3, 4], [5, 6]]
        input_2 = [[7, 8], [9, 10], [11, 12]]
        inputs = tf.constant([input_1, input_2], dtype=tf.float32)

        expected = [[3, 4], [9, 10]]

        with self.test_session():
            self.assertAllEqual(layer(inputs).eval(), expected)

    def test_call_with_list(self):
        layer = FixedMeanPool(name='call_with_list')
        input_1 = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
        input_2 = tf.constant(
            [[7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.float32)
        inputs = [input_1, input_2]

        expected = [[3, 4], [11, 12]]

        with self.test_session():
            self.assertAllEqual(layer(inputs).eval(), expected)
