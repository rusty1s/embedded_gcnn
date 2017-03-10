import tensorflow as tf

from .model import Model
from ..graph.adjacency import grid_adj, normalize_adj, invert_adj, adj_to_tf
from ..graph.laplacian import laplacian, lmax, rescale_lap
from ..layer.chebyshev_gcnn import ChebyshevGCNN
from ..layer.fc import FC


class MNISTChebyshevGCNN(Model):
    def __init__(self, normalized=True, **kwargs):
        # 3 placeholders: features, labels, dropout
        super(MNISTChebyshevGCNN, self).__init__(**kwargs)

        # We use the same graph for every example.
        adj = grid_adj([28, 28], connectivity=8)
        adj = normalize_adj(adj)
        adj = invert_adj(adj)
        lap = laplacian(adj, normalized)
        lap = rescale_lap(lap, lmax(lap))
        self.lap = adj_to_tf(lap)

        self.build()

    def _preprocess(self):
        return tf.reshape(self.inputs, [-1, 28 * 28, 1])

    def _build(self):
        chebyshev_gcnn1 = ChebyshevGCNN(
            1, 8, self.lap, max_degree=5, logging=self.logging)
        fc1 = FC(28 * 28 * 8, 1024, logging=self.logging)
        fc2 = FC(1024,
                 10,
                 dropout=self.placeholders['dropout'],
                 act=lambda x: x,
                 logging=self.logging)

        self.layers = [chebyshev_gcnn1, fc1, fc2]
