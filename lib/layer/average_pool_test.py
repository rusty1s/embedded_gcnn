import tensorflow as tf

from .average_pool import AveragePool


class AveragePoolTest(tf.test.TestCase):
    def test_init(self):
        layer = AveragePool()
        self.assertEqual(layer.name, 'averagepool_1')

    def test_call_rank3(self):
        layer = AveragePool(name='call_rank3')
        input_1 = [[1, 2], [3, 4], [5, 6], [7, 8]]
        input_2 = [[9, 10], [11, 12], [13, 14], [15, 16]]
        inputs = tf.constant([input_1, input_2], dtype=tf.float32)

        expected = [[4, 5], [12, 13]]

        with self.test_session():
            self.assertAllEqual(layer(inputs).eval(), expected)

    def test_call_rank4(self):
        layer = AveragePool(name='call_rank4')
        input_1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        input_2 = [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
        inputs = tf.constant([input_1, input_2], dtype=tf.float32)

        expected = [[4, 5], [12, 13]]

        with self.test_session():
            self.assertAllEqual(layer(inputs).eval(), expected)

    def test_call_other(self):
        layer = AveragePool(name='call_other')
        inputs = [[1, 2], [3, 4]]
        inputs = tf.constant(inputs)

        with self.assertRaises(AssertionError):
            layer(inputs)
