import tensorflow as tf

from .image_augment import ImageAugment


class ImageAugmentTest(tf.test.TestCase):
    def test_init(self):
        layer = ImageAugment()
        self.assertEqual(layer.name, 'imageaugment_1')

    def test_call(self):
        layer = ImageAugment(name='call')
        inputs = tf.constant([[[
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.5],
        ], [
            [0.5, 0.6, 0.7],
            [0.7, 0.8, 0.9],
        ]], [[
            [0.1, 0.2, 0.3],
            [0.3, 0.4, 0.5],
        ], [
            [0.5, 0.6, 0.7],
            [0.7, 0.8, 0.9],
        ]]])

        outputs = layer(inputs)

        with self.test_session():
            # Augmentation is random and therefore not testable, so we just ran
            # it and ensure that the computation succeeds.
            self.assertEqual(outputs.eval().shape, (2, 2, 2, 3))
