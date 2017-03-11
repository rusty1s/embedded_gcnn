from __future__ import division

import numpy as np

from .model import Model
from ..graph.adjacency import grid_adj, normalize_adj, invert_adj, adj_to_tf
from ..graph.coarsening import coarsen_adj
from ..graph.laplacian import laplacian, lmax, rescale_lap
from ..layer.chebyshev_gcnn import ChebyshevGCNN as Conv
from ..layer.max_pool_gcnn import MaxPoolGCNN as MaxPool
from ..layer.fc import FC


class MNISTChebyshevGCNN(Model):
    def __init__(self, normalized=True, **kwargs):
        # 3 placeholders: features, labels, dropout
        super(MNISTChebyshevGCNN, self).__init__(**kwargs)

        # We use the same graph for every example.
        adj = grid_adj([28, 28], connectivity=8)
        adj = normalize_adj(adj)
        adj = invert_adj(adj)

        adjs, perm = coarsen_adj(adj, levels=4, rid=np.arange(28 * 28))
        self.perm = perm

        self.laps = []
        for adj in [adjs[0], adjs[2]]:
            lap = laplacian(adj, normalized)
            lap = rescale_lap(lap, lmax(lap, normalized))
            self.laps.append(adj_to_tf(lap))

        self.build()

    def _build(self):

        conv_1 = Conv(1, 32, self.laps[0], max_degree=2, logging=self.logging)
        max_pool_1 = MaxPool(size=4, logging=self.logging)
        conv_2 = Conv(32, 64, self.laps[1], max_degree=2, logging=self.logging)
        max_pool_2 = MaxPool(size=4, logging=self.logging)
        fc_1 = FC((976 // 4 // 4) * 64, 1024, logging=self.logging)
        fc_2 = FC(1024,
                  10,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [conv_1, max_pool_1, conv_2, max_pool_2, fc_1, fc_2]
