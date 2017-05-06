import tensorflow as tf

from .layer import Layer


class LayerTest(tf.test.TestCase):
    def test_init(self):
        layer = Layer(name='layer')
        self.assertEqual(layer.name, 'layer')
        self.assertEqual(layer.logging, False)

        layer = Layer(logging=True)
        self.assertEqual(layer.name, 'layer_1')
        self.assertEqual(layer.logging, True)

        layer = Layer()
        self.assertEqual(layer.name, 'layer_2')

    def test_call(self):
        layer = Layer(name='call', logging=True)
        inputs = tf.constant([1, 2, 3])

        with self.test_session():
            self.assertAllEqual(layer(inputs).eval(), inputs.eval())
