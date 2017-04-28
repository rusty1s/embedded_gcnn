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
        layer = Layer(name='call')
        self.assertRaises(NotImplementedError, layer.__call__, None)
