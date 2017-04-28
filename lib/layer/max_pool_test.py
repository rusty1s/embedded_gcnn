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

    def test_call(self):
        layer = MaxPool(size=2, name='call')
        input_1 = [[1, 2], [3, 4], [5, 6]]
        input_2 = [[7, 8], [9, 10], [11, 12]]
        input_1 = tf.constant(input_1, dtype=tf.float32)
        input_2 = tf.constant(input_2, dtype=tf.float32)
        inputs = [input_1, input_2]
        outputs = layer(inputs)

        expected = [[[3, 4], [5, 6]], [[9, 10], [11, 12]]]

        with self.test_session():
            self.assertEqual(len(outputs), 2)
            self.assertAllEqual(outputs[0].eval(), expected[0])
            self.assertAllEqual(outputs[1].eval(), expected[1])
