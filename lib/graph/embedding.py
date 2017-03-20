from __future__ import division

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


def points_to_embedded(points, adj, dtype=np.float32):
    """Builds an embedded adjacency matrix based on points of nodes."""

    # Initialize helper variables.
    n = adj.shape[0]
    rows, cols, _ = sp.find(adj)  # Ignore edge weights of `adj`.
    perm = np.argsort(rows)
    rows = rows[perm]
    cols = cols[perm]
    nnz = rows.shape[0]
    dists = np.empty((nnz), dtype=dtype)
    rads = np.empty((nnz), dtype=dtype)

    for i in xrange(nnz):
        # Calculate distance and angle of each edge vector.
        vector = points[cols[i]] - points[rows[i]]
        dists[i] = np.sum(np.power(vector, 2))
        rad = np.arctan2(vector[0], vector[1])
        rads[i] = rad if rad > 0 else rad + 2 * np.pi

    adj_dist = sp.coo_matrix((dists, (rows, cols)), (n, n))
    adj_rad = sp.coo_matrix((rads, (rows, cols)), (n, n))

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
