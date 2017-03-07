import tensorflow as tf

from .chebyshev_gcnn import ChebyshevGCNN


class ChebyshevGCNNTest(tf.test.TestCase):
    def test_init(self):
        layer = ChebyshevGCNN(1, 2, max_degree=3)
        self.assertEqual(layer.name, 'chebyshevgcnn_1')
        self.assertEqual(layer.act, tf.nn.relu)
        self.assertEqual(layer.bias, True)
        self.assertEqual(layer.logging, False)
        self.assertIn('weights', layer.vars)
        self.assertEqual(layer.vars['weights'].get_shape(), [3, 1, 2])
        self.assertIn('bias', layer.vars)
        self.assertEqual(layer.vars['bias'].get_shape(), [2])

        layer = ChebyshevGCNN(3, 4, max_degree=5, bias=False, logging=True)
        self.assertEqual(layer.name, 'chebyshevgcnn_2')
        self.assertEqual(layer.act, tf.nn.relu)
        self.assertEqual(layer.bias, False)
        self.assertEqual(layer.logging, True)
        self.assertIn('weights', layer.vars)
        self.assertEqual(layer.vars['weights'].get_shape(), [5, 3, 4])
        self.assertNotIn('bias', layer.vars)
