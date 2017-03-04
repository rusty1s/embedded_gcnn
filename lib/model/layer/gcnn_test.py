import tensorflow as tf

from .gcnn import diag, normalize_adj


class GraphConvolutionTest(tf.test.TestCase):
    def test_diag(self):
        d = diag(tf.constant([1, 2, 3]))
        expected = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

        with self.test_session():
            d = tf.sparse_tensor_to_dense(d)
            self.assertAllEqual(d.eval(), expected)

    def test_normalize_adj(self):
        adj = tf.SparseTensor(
            indices=[[0, 1], [1, 0], [1, 2], [2, 1]],
            values=[1, 1, 1, 1],
            dense_shape=[3, 3])
        expected = [[1, 1, 0], [1, 1, 1], [0, 1, 1]]

        with self.test_session():
            adj = normalize_adj(adj)
            adj = tf.sparse_tensor_to_dense(adj)
            self.assertAllEqual(adj.eval(), expected)
