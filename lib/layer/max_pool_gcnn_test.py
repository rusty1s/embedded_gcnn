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

    # def test_call_with_list(self):
    #     layer = MaxPoolGCNN(size=2, name='call_with_list')
    #     input_1 = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    #     input_2 = tf.constant(
    #         [[7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.float32)
    #     outputs = layer([input_1, input_2])
    #     self.assertEqual(len(outputs), 2)

    #     with self.test_session():
    #         self.assertAllEqual(outputs[0].eval(), [[3, 4], [5, 6]])
    #         self.assertAllEqual(outputs[1].eval(),
    #                             [[9, 10], [13, 14], [15, 16]])
