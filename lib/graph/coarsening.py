from __future__ import division
from six.moves import xrange

import numpy as np
import numpy_groupies as npg
import scipy.sparse as sp

from .clustering import normalized_cut
from .distortion import perm_adj
from .adjacency import points_to_adj


def coarsen_adj(adj,
                points,
                mass,
                levels,
                scale_invariance=False,
                stddev=1,
                rid=None):

    # Coarse adjacency  a defined number of levels deep.
    adjs_dist, adjs_rad, cluster_maps = _coarsen_adj(
        adj, points, mass, levels, scale_invariance, stddev, rid)

    # Permutate adjacencies to a binary tree for an efficient O(n) pooling.
    perms = _compute_perms(cluster_maps)
    adjs_dist = [perm_adj(adjs_dist[i], perms[i]) for i in xrange(levels + 1)]
    adjs_rad = [perm_adj(adjs_rad[i], perms[i]) for i in xrange(levels + 1)]

    return adjs_dist, adjs_rad, perms[0]


def _coarsen_adj(adj,
                 points,
                 mass,
                 levels,
                 scale_invariance=False,
                 stddev=1,
                 rid=None):

    adj_dist, adj_rad = points_to_adj(adj, points, scale_invariance, stddev)

    adjs_dist = [adj_dist]
    adjs_rad = [adj_rad]
    cluster_maps = []

    for _ in xrange(levels):
        # Calculate normalized cut clustering.
        cluster_map = normalized_cut(adj_dist, rid)
        cluster_maps.append(cluster_map)

        # Coarsen adjacency.
        adj, points, mass = _coarsen_clustered_adj(adj, points, mass,
                                                   cluster_map)

        # Compute to distance/radian adjacency.
        adj_dist, adj_rad = points_to_adj(adj, points, scale_invariance,
                                          stddev)
        adjs_dist.append(adj_dist)
        adjs_rad.append(adj_rad)

        # Iterate by degree at next iteration.
        if adj_dist.data.size > 0:
            degree = npg.aggregate(adj_dist.row, adj_dist.data, func='sum')
            rid = np.argsort(degree)
        else:
            rid = None

    return adjs_dist, adjs_rad, cluster_maps


def _coarsen_clustered_adj(adj, points, mass, cluster_map):
    rows = cluster_map[adj.row]
    cols = cluster_map[adj.col]

    n = cluster_map.max() + 1
    adj = sp.coo_matrix((adj.data, (rows, cols)), shape=(n, n))
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj = adj.tocsc().tocoo()  # Sum up duplicate row/col entries.

    points_y = mass * points[:, :1].flatten()
    points_y = npg.aggregate(cluster_map, points_y, func='sum')

    points_x = mass * points[:, 1:].flatten()
    points_x = npg.aggregate(cluster_map, points_x, func='sum')

    mass = npg.aggregate(cluster_map, mass, func='sum')

    points_y = points_y / mass
    points_x = points_x / mass

    points_y = np.reshape(points_y, (-1, 1))
    points_x = np.reshape(points_x, (-1, 1))
    points = np.concatenate((points_y, points_x), axis=1)

    return adj.tocsc().tocoo(), points, mass


def _compute_perms(cluster_maps):
    max_cluster = np.max(cluster_maps[-1]) + 1
    n = max_cluster
    perm = np.arange(n)
    perms = [perm]

    for i in xrange(len(cluster_maps) - 1, -1, -1):  # Iterate backwards
        cluster_map = cluster_maps[i]

        # Add single fake nodes to end of cluster map.
        idx, counts = np.unique(cluster_map, return_counts=True)
        singles = idx[np.where(counts == 1)]

        # Add double fake nodes to end of cluster map.
        max_cluster = (cluster_map.size + singles.size) // 2
        doubles = np.arange(max_cluster, n).repeat(2)

        cluster_map = np.concatenate((cluster_map, singles, doubles))

        # Permutate cluster_map.
        cluster_map = np.argsort(perm)[cluster_map]
        rid = np.argsort(cluster_map)
        n *= 2
        perm = np.arange(n)[rid]
        perms.append(perm)

    return perms[::-1]
