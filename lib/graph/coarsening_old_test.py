from unittest import TestCase

# import numpy as np
# from numpy.testing import *
import scipy.sparse as sp

# from .coarsening import cluster_adj
from .coarsening_old import coarsen


class CoarseningTest(TestCase):
    def test_cluster_adj(self):
        adj = [[0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 2], [0, 1, 2, 0]]
        adj = sp.coo_matrix(adj)

        graphs, perm = coarsen(adj, levels=2)
        print('graphs', graphs)
        print('perm', perm)
        # print(cluster_adj(adj))

        # adj = [[0, 3, 2, 0, 0], [3, 0, 0, 2, 0], [2, 0, 0, 3, 0],
        #        [0, 2, 3, 0, 1], [0, 0, 0, 1, 0]]
        # adj = sp.coo_matrix(adj)
        # print(cluster_adj(adj))
