from __future__ import division

import numpy as np
import tensorflow as tf

from .model import Model
from ..graph.adjacency import grid_adj, normalize_adj, invert_adj, adj_to_tf
from ..graph.coarsening import coarsen_adj
from ..graph.preprocess import preprocess_adj
from ..layer.gcnn import GCNN as Conv
from ..layer.max_pool_gcnn import MaxPoolGCNN as MaxPool
from ..layer.fc import FC


class MNIST_GCNN(Model):
    def __init__(self, **kwargs):
        # 3 placeholders: features, labels, dropout
        super(MNIST_GCNN, self).__init__(**kwargs)

        # We use the same graphs for every example.
        adj = grid_adj([28, 28], connectivity=8)
        adj = normalize_adj(adj)
        adj = invert_adj(adj)

        # adjs, perm = coarsen_adj(adj, levels=4, rid=np.arange(28 * 28))
        # self.perm = perm

        # self.adjs = []
        # for adj in [adjs[0], adjs[2]]:
        #     adj = preprocess_adj(adj)
        #     self.adjs.append(adj_to_tf(adj))

        adj = preprocess_adj(adj)
        self.adj = adj_to_tf(adj)

        self.build()

    def _build(self):
        conv_1 = Conv(1, 32, self.adj, logging=self.logging)
        # max_pool_1 = MaxPool(size=4, logging=self.logging)
        # conv_2 = Conv(32, 64, self.adjs[1], logging=self.logging)
        # max_pool_2 = MaxPool(size=4, logging=self.logging)
        fc_1 = FC(28 * 28 * 32, 1024, logging=self.logging)
        fc_2 = FC(1024,
                  10,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [conv_1, fc_1, fc_2]
        # self.layers = [conv_1, max_pool_1, conv_2, max_pool_2, fc_1, fc_2]
