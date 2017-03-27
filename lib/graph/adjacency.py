from __future__ import division

from six.moves import xrange

import numpy as np
import scipy.sparse as sp


def normalize_adj(adj, locale=False):
    """Normalize adjacency matrix to interval [0, 1]."""
    if not locale:
        return (1 / adj.max()) * adj
    else:
        max_row = 1 / adj.max(axis=1).toarray().flatten()
        diag = sp.diags(max_row)
        adj = adj.dot(diag)
        return (adj + adj.transpose()) / 2


def invert_adj(m, stddev=1):
    """Return (inverted) gaussian kernel representation."""

    if sp.issparse(m):
        m = m.copy()
        m.data = np.exp(-m.data / (2 * stddev * stddev))
        return m
    else:
        return np.exp(-m / (2 * stddev * stddev))


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
    """Return whether the node has a top, bottom, left and right neighbor."""
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
