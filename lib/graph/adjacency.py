from __future__ import division

import numpy as np
import scipy.sparse as sp
import numpy_groupies as npg


def zero_one_scale_adj(adj, scale_invariance=False):
    """Normalize adjacency matrix to interval [0, 1]. Can choose between global
    or locale normalization."""

    data = adj.data

    if data.size == 0:
        return adj

    if not scale_invariance:
        adj.data = (1 / data.max()) * data
    else:
        rows = adj.row
        multiplicator = 1 / npg.aggregate(rows, data, func='max')
        multiplicator = multiplicator[rows]
        adj.data = data * multiplicator
        return (adj + adj.transpose()) / 2

    return adj


def invert_adj(adj, stddev=1):
    """Return (inverted) gaussian kernel representation."""

    denominator = -2 * stddev * stddev
    adj.data = np.exp(adj.data / denominator)
    return adj


def points_to_l2_adj(adj, points):
    """Build an embedded adjacency matrix based on points (y, x) of nodes."""

    ys = points[:, :1].flatten()
    xs = points[:, 1:].flatten()

    rows = adj.row
    rows_ys = ys[rows]
    rows_xs = xs[rows]

    cols = adj.col
    cols_ys = ys[cols]
    cols_xs = xs[cols]

    vector_y = rows_ys - cols_ys
    vector_x = cols_xs - rows_xs

    dists = vector_y * vector_y + vector_x * vector_x
    rads = np.arctan2(vector_x, vector_y)
    # Adjust radians to lay in ]0, 2 * pi].
    rads = np.where(rads > 0, rads, rads + 2 * np.pi)

    n = adj.shape[0]
    adj_dist = sp.coo_matrix((dists, (rows, cols)), (n, n))
    adj_rad = sp.coo_matrix((rads, (rows, cols)), (n, n))

    return adj_dist, adj_rad


def points_to_adj(adj, points, scale_invariance=False, stddev=1):
    """Build an embedded adjacency matrix based on points (y, x) of nodes, so
    that nearer nodes have a smaller weight."""

    adj_dist, adj_rad = points_to_l2_adj(adj, points)
    adj_dist = zero_one_scale_adj(adj_dist, scale_invariance)
    adj_dist = invert_adj(adj_dist, stddev)
    return adj_dist, adj_rad
