import tensorflow as tf

from .model import Model
from ..graph.adjacency import grid_adj, normalize_adj, invert_adj, adj_to_tf
from ..graph.preprocess import preprocess_adj
from ..layer.gcnn import GCNN
from ..layer.fc import FC


class MNIST_GCNN(Model):
    def __init__(self, **kwargs):
        # 3 placeholders: features, labels, dropout
        super(MNIST_GCNN, self).__init__(**kwargs)

        adj = grid_adj([28, 28], connectivity=8)
        adj = normalize_adj(adj)
        adj = invert_adj(adj)
        adj = preprocess_adj(adj)
        self.adj = adj_to_tf(adj)

        self.build()

    def _preprocess(self):
        return tf.reshape(self.inputs, [-1, 28 * 28, 1])

    def _build(self):
        gcnn1 = GCNN(1, 32, adjs=self.adj, logging=self.logging)
        fc1 = FC(28 * 28 * 32, 1024, logging=self.logging)
        fc2 = FC(1024,
                 10,
                 dropout=self.placeholders['dropout'],
                 act=lambda x: x,
                 logging=self.logging)

        self.layers = [gcnn1, fc1, fc2]
