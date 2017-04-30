from six.moves import xrange

import numpy as np
import scipy.sparse as sp


def grid_points(shape, dtype=np.float32):
    x, y = np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0]))
    x = x.flatten()
    y = np.flip(y.flatten(), axis=0)

    z = np.empty((shape[0] * shape[1], 2), dtype)
    z[:, 0] = x
    z[:, 1] = y
    return z


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

    return adj.tocoo()


def _grid_neighbors(v, height, width):
    """Return whether the node has a top, right, bottom and right neighbor."""

    top = v >= width
    bottom = v < (height - 1) * width
    left = v % width
    right = v % width < width - 1

    return top, right, bottom, left


def _grid_adj_4(adj, height, width):
    """Add edges to vertical/horizontal nodes on grid adjacency."""

    for v in xrange(height * width):
        top, right, bottom, left = _grid_neighbors(v, height, width)

        if top:
            adj[v, v - width] = 1
        if right:
            adj[v, v + 1] = 1
        if bottom:
            adj[v, v + width] = 1
        if left:
            adj[v, v - 1] = 1

    return adj


def _grid_adj_8(adj, height, width):
    """Add edges to diagonal nodes on grid adjacency."""

    for v in xrange(height * width):
        top, right, bottom, left = _grid_neighbors(v, height, width)

        if top and right:
            adj[v, v - width + 1] = 2
        if bottom and right:
            adj[v, v + width + 1] = 2
        if bottom and left:
            adj[v, v + width - 1] = 2
        if top and left:
            adj[v, v - width - 1] = 2

    return adj
