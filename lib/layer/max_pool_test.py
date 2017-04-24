import tensorflow as tf

from .max_pool import MaxPool


class MaxPool2DTest(tf.test.TestCase):
    def test_init(self):
        layer = MaxPool(size=2, stride=1)
        self.assertEqual(layer.name, 'maxpool_1')
        self.assertEqual(layer.size, 2)
        self.assertEqual(layer.stride, 1)

        layer = MaxPool(size=2)
        self.assertEqual(layer.size, 2)
        self.assertEqual(layer.stride, 2)

    def test_call_rank3(self):
        layer = MaxPool(size=2, name='call_rank3')
        input_1 = [[1, 2], [3, 4], [5, 6]]
        input_2 = [[7, 8], [9, 10], [11, 12]]
        inputs = tf.constant([input_1, input_2], dtype=tf.float32)

        expected = [[[3, 4], [5, 6]], [[9, 10], [11, 12]]]

        with self.test_session():
            self.assertAllEqual(layer(inputs).eval(), expected)

    def test_call_rank4(self):
        layer = MaxPool(size=2, name='call_rank4')
        input_1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        input_2 = [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
        inputs = tf.constant([input_1, input_2], dtype=tf.float32)

        expected = [[[[7, 8]]], [[[15, 16]]]]

        with self.test_session():
            self.assertAllEqual(layer(inputs).eval(), expected)

    def test_call_other(self):
        layer = MaxPool(size=2, name='call_rank4')
        inputs = [[1, 2], [3, 4]]
        inputs = tf.constant(inputs)

        with self.assertRaises(AssertionError):
            layer(inputs)
