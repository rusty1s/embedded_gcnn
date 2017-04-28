import tensorflow as tf

from .var_layer import VarLayer


class VarLayerTest(tf.test.TestCase):
    def test_init(self):
        layer = VarLayer(weight_shape=[1, 2], bias_shape=[2])
        self.assertEqual(layer.name, 'varlayer_1')
        self.assertTrue(layer.bias)
        self.assertEqual(layer.act, tf.nn.relu)
        self.assertEqual(layer.vars['weights'].get_shape(), [1, 2])
        self.assertEqual(tf.get_collection('losses'), [])
        self.assertEqual(layer.vars['bias'].get_shape(), [2])

        layer = VarLayer(
            weight_shape=[1, 2],
            bias_shape=[2],
            weight_stddev=0.0,
            weight_decay=0.01,
            bias=True,
            bias_constant=1)
        losses = tf.get_collection('losses')

        expected_weights = [[0, 0]]
        expected_bias = [1, 1]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            self.assertEqual(layer.name, 'varlayer_2')
            self.assertTrue(layer.bias)
            self.assertAllEqual(layer.vars['weights'].eval(), expected_weights)
            self.assertAllEqual(layer.vars['bias'].eval(), expected_bias)
            self.assertEqual(losses[0].name, 'varlayer_2_vars/weight_loss:0')
            self.assertEqual(losses[0].eval(), 0)

        layer = VarLayer(weight_shape=[1, 2], bias_shape=[2], bias=False)
        self.assertEqual(layer.name, 'varlayer_3')
        self.assertFalse(layer.bias)
        self.assertIsNone(layer.vars.get('bias', None))
