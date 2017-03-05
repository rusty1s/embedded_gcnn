from __future__ import division

from six.moves import xrange

import numpy as np
import scipy.sparse as sp


def normalize_adj(adj):
    """Normalize adjacency matrix to interval [0, 1]."""
    return (1 / adj.max()) * adj


def invert_adj(m, sigma=1):
    """Return (inverted) gaussian kernel representation."""

    if sp.issparse(m):
        m = m.copy()
        m.data = np.exp(-m.data / (2 * sigma * sigma))
        return m
    else:
        return np.exp(-m / (2 * sigma * sigma))


def _grid_adj_4(adj, height, width):
    for i in xrange(height * width):
        if i % width > 0:
            adj[i, i-1] = 1
        if i % width < width-1:
            adj[i, i+1] = 1
        if i >= width:
            adj[i, i-width] = 1
        if i < (height-1) * width:
            adj[i, i+width] = 1
    return adj


def _grid_adj_8(adj, height, width):
    for i in xrange(height * width):
        if i >= width and i % width > 0:
            adj[i, i-width-1] = 2
        if i >= width and i % width < width-1:
            adj[i, i-width+1] = 2
        if i < (height-1) * width and i % width > 0:
            adj[i, i+width-1] = 2
        if i < (height-1) * width and i % width < width-1:
            adj[i, i+width+1] = 2
    return adj


def grid_adj(shape, connectivity=4, dtype=np.float32):
    """Return adjacency matrix of a regular grid."""

    assert connectivity == 4 or connectivity == 8,\
        'Invalid connectivity {}'.format(connectivity)

    height, width = shape
    n = height * width
    adj = sp.lil_matrix((n, n), dtype=dtype)

    adj = _grid_adj_4(adj, height, width)

    if connectivity == 8:
        adj = _grid_adj_8(adj, height, width)

    return adj



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

    return adj
