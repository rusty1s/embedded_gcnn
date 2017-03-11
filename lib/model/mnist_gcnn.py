import tensorflow as tf

from .model import Model
from ..graph.adjacency import grid_adj, normalize_adj, invert_adj, adj_to_tf
from ..graph.preprocess import preprocess_adj
from ..layer.gcnn import GCNN as Conv
from ..layer.fc import FC


class MNIST_GCNN(Model):
    def __init__(self, **kwargs):
        # 3 placeholders: features, labels, dropout
        super(MNIST_GCNN, self).__init__(**kwargs)

        # We use the same graph for every example.
        adj = grid_adj([28, 28], connectivity=8)
        adj = normalize_adj(adj)
        adj = invert_adj(adj)
        adj = preprocess_adj(adj)
        self.adj = adj_to_tf(adj)

        self.build()

    def _preprocess(self):
        return tf.reshape(self.inputs, [-1, 28 * 28, 1])

    def _build(self):
        conv_1 = Conv(1, 8, self.adj, logging=self.logging)
        fc_1 = FC(28 * 28 * 8, 1024, logging=self.logging)
        fc_2 = FC(1024,
                  10,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [conv_1, fc_1, fc_2]
