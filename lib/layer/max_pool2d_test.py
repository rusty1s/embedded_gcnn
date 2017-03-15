import tensorflow as tf

from .max_pool2d import MaxPool2d


class MaxPool2DTest(tf.test.TestCase):
    def test_init(self):
        layer = MaxPool2d(size=2, stride=1)
        self.assertEqual(layer.name, 'maxpool2d_1')
        self.assertEqual(layer.logging, False)
        self.assertEqual(layer.size, 2)
        self.assertEqual(layer.stride, 1)

    def test_call(self):
        layer = MaxPool2d(size=2, stride=2, name='call')
        input_1 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        input_2 = [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
        inputs = tf.constant([input_1, input_2], dtype=tf.float32)

        expected = [[[[7, 8]]], [[[15, 16]]]]

        with self.test_session():
            self.assertAllEqual(layer(inputs).eval(), expected)
