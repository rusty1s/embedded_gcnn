from __future__ import division

import numpy as np
import networkx as nx
from skimage.future.graph import RAG


def segmentation_adjacency(segmentation, connectivity=2, dtype=np.float32):
    graph = RAG(segmentation, connectivity)
    adj = nx.to_scipy_sparse_matrix(
        graph, dtype=dtype, weight=None, format='coo')

    n = len(graph)
    mass = np.zeros((n), dtype)
    centroid = np.zeros((n, 2), dtype)

    for index in np.ndindex(segmentation.shape):
        idx = segmentation[index]
        mass[idx] += 1
        centroid[idx] += index

    return centroid / mass, adj, mass
