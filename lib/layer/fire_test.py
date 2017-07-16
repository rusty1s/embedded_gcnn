import tensorflow as tf

from .fire import Fire


class FireTest(tf.test.TestCase):
    def test_call(self):
        layer = Fire(1, 1, 4)
        self.assertEqual(layer.name, 'fire_1')

        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        inputs = tf.constant([image, image], tf.float32)
        inputs = tf.reshape(inputs, [2, 3, 3, 1])

        outputs = layer(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(outputs.eval().shape, (2, 3, 3, 8))
