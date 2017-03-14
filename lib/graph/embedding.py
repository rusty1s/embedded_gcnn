from six.moves import xrange

import numpy as np
import scipy.sparse as sp


from .adjacency import _grid_neighbors


def grid_embedded_adj(shape, connectivity=4, dtype=np.float32):
    assert connectivity == 4 or connectivity == 8,\
        'Invalid connectivity {}'.format(connectivity)

    height, width = shape
    n = height * width
    adj_dist = sp.lil_matrix((n, n), dtype=dtype)
    adj_rad = sp.lil_matrix((n, n), dtype=np.float32)

    adj_dist, adj_rad = _grid_adj_4(adj_dist, adj_rad, height, width)

    if connectivity == 8:
        adj_dist, adj_rad = _grid_adj_8(adj_dist, adj_rad, height, width)

    return adj_dist.tocoo(), adj_rad.tocoo()


def _grid_adj_4(adj_dist, adj_rad, height, width):
    for v in xrange(height * width):
        top, right, bottom, left = _grid_neighbors(v, height, width)

        if top:
            adj_dist[v, v - width] = 1
            adj_rad[v, v - width] = 2 * np.pi
        if right:
            adj_dist[v, v + 1] = 1
            adj_rad[v, v + 1] = 0.5 * np.pi
        if bottom:
            adj_dist[v, v + width] = 1
            adj_rad[v, v + width] = np.pi
        if left:
            adj_dist[v, v - 1] = 1
            adj_rad[v, v - 1] = 1.5 * np.pi

    return adj_dist, adj_rad


def _grid_adj_8(adj_dist, adj_rad, height, width):
    for v in xrange(height * width):
        top, right, bottom, left = _grid_neighbors(v, height, width)

        if top and right:
            adj_dist[v, v - width + 1] = 2
            adj_rad[v, v - width + 1] = 0.25 * np.pi
        if bottom and right:
            adj_dist[v, v + width + 1] = 2
            adj_rad[v, v + width + 1] = 0.75 * np.pi
        if bottom and left:
            adj_dist[v, v + width - 1] = 2
            adj_rad[v, v + width - 1] = 1.25 * np.pi
        if top and left:
            adj_dist[v, v - width - 1] = 2
            adj_rad[v, v - width - 1] = 1.75 * np.pi

    return adj_dist, adj_rad
