import tensorflow as tf

from .average_pool import AveragePool


class AveragePoolTest(tf.test.TestCase):
    def test_init(self):
        layer = AveragePool()
        self.assertEqual(layer.name, 'averagepool_1')

    def test_call(self):
        layer = AveragePool(name='call')
        input_1 = [[1, 2], [3, 4], [5, 6], [7, 8]]
        input_2 = [[9, 10], [11, 12], [13, 14], [15, 16]]
        input_1 = tf.constant(input_1, dtype=tf.float32)
        input_2 = tf.constant(input_2, dtype=tf.float32)
        inputs = [input_1, input_2]
        outputs = layer(inputs)

        expected = [[4, 5], [12, 13]]

        with self.test_session():
            # Average pooling converts lists to tensors.
            self.assertAllEqual(outputs.eval(), expected)
