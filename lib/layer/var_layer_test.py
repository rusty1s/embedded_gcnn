import tensorflow as tf

from .var_layer import VarLayer


class VarLayerTest(tf.test.TestCase):
    def test_init(self):
        layer = VarLayer(weight_shape=[1, 2], bias_shape=[2])
        self.assertEqual(layer.name, 'varlayer_1')
        self.assertTrue(layer.bias)
        self.assertEqual(layer.act, tf.nn.relu)
        self.assertEqual(layer.vars['weights'].get_shape(), [1, 2])
        self.assertEqual(layer.vars['bias'].get_shape(), [2])
