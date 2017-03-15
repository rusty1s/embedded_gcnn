import tensorflow as tf

from .max_pool_gcnn import MaxPoolGCNN


class MaxPoolGCNNTest(tf.test.TestCase):
    def test_init(self):
        layer = MaxPoolGCNN(size=2)
        self.assertEqual(layer.name, 'maxpoolgcnn_1')
        self.assertEqual(layer.logging, False)
        self.assertEqual(layer.size, 2)

    def test_call(self):
        layer = MaxPoolGCNN(size=2, name='call')
        input_1 = [[1, 2], [3, 4], [5, 6]]
        input_2 = [[7, 8], [9, 10], [11, 12]]
        inputs = tf.constant([input_1, input_2], dtype=tf.float32)

        expected = [[[3, 4], [5, 6]], [[9, 10], [11, 12]]]

        with self.test_session():
            self.assertAllEqual(layer(inputs).eval(), expected)
