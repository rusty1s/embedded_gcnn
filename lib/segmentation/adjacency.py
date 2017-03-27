from six.moves import xrange

import numpy as np
import networkx as nx
from skimage.future.graph import RAG
from skimage.measure import regionprops


def segmentation_adjacency(segmentation, connectivity=2, dtype=np.float32):
    graph = RAG(segmentation, connectivity)
    adj = nx.to_scipy_sparse_matrix(
        graph, dtype=dtype, weight=None, format='coo')

    props = regionprops(segmentation + 1)
    n = len(props)
    points = np.array(
        [np.flip(props[i]['centroid'], axis=0) for i in xrange(n)], dtype)
    mass = np.array([props[i]['area'] for i in xrange(n)], dtype)

    return points, adj, mass
