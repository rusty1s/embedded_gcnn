import numpy as np
import scipy.sparse as sp


def embedded_adj(points, neighbors, dtype=np.float32):
    """Return adjacency matrix of an embedded graph."""

    num_nodes = points.shape[0]
    adj = sp.lil_matrix((num_nodes, num_nodes), dtype=dtype)

    for v1, v2 in neighbors:
        p1 = points[v1]
        p2 = points[v2]
        d = np.abs(p1[0] - p2[0])**2 + np.abs(p1[1] - p2[1])**2
        adj[v1, v2] = d
        adj[v2, v1] = d

    return adj.tocoo()
