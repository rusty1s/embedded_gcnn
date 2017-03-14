from __future__ import division

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


def partition_embedded_adj(adj_dist, adj_rad, num_partitions, offset=0.0):
    adj_rad = adj_rad.tocoo()

    n = adj_dist.shape[0]
    dists = adj_dist.data
    rads = adj_rad.data.copy()
    rows = adj_rad.row
    cols = adj_rad.col
    adjs = []

    # Take care of offset greater interval case.
    interval = 2 * np.pi / num_partitions
    offset = offset % interval
    max_rad = offset

    for i in xrange(num_partitions+1):
        if i < num_partitions:
            adj = sp.coo_matrix((n, n), dtype=adj_dist.dtype)
            adjs.append(adj)
        else:
            adj = adjs[0]

        indices = np.where(rads < max_rad)
        rads[indices] = np.inf

        adj.row = np.concatenate((adj.row, rows[indices]))
        adj.col = np.concatenate((adj.col, cols[indices]))
        adj.data = np.concatenate((adj.data, dists[indices]))

        # Be sure to take all indices in the last step.
        max_rad = max_rad + interval if i < num_partitions - 1 else 7

    return adjs
