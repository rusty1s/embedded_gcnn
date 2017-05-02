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
        degree = npg.aggregate(adj_dist.row, adj_dist.data, func='sum')
        rid = np.argsort(degree)

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
    # Last permutation is the ordered list of the number of clusters in the
    # last cluster map.
    n = np.max(cluster_maps[-1]) + 1
    perm = np.arange(n)
    perms = [perm]

    # Iterate backwards through cluster_maps.
    for i in xrange(len(cluster_maps) - 1, -1, -1):
        cluster_map = cluster_maps[i]
        cur_singleton_idx = cluster_map.size

        perm_last = perm
        perm = np.zeros((2 * perm_last.size), perm_last.dtype)
        for j in xrange(perm_last.size):
            # Indices of the cluster map that correspond to the calculated
            # permutation.
            nodes_idx = np.where(cluster_map == perm_last[j])[0]

            # Add fake nodes if neccassary.
            if nodes_idx.size == 1:
                perm[2*j] = nodes_idx[0]
                perm[2*j+1] = cur_singleton_idx
                cur_singleton_idx += 1
            elif nodes_idx.size == 0:
                perm[2*j] = cur_singleton_idx
                perm[2*j+1] = cur_singleton_idx + 1
                cur_singleton_idx += 2
            else:
                perm[2*j] = nodes_idx[0]
                perm[2*j+1] = nodes_idx[1]

        perms.append(perm)

    # Reverse permutations.
    return perms[::-1]


def _compute_perms2(cluster_maps):
    n = np.max(cluster_maps[-1]) + 1
    perm = np.arange(n)
    perms = [perm]

    for i in xrange(len(cluster_maps) - 1, -1, -1):
        last_perm = perm
        cluster_map = cluster_maps[i]
        n = 2 * n

        idx, counts = np.unique(cluster_map, return_counts=True)
        rid = np.where(counts == 1)[0]
        perm = np.concatenate((cluster_map, idx[rid]), axis=0)
        m = (n - perm.size) // 2
        bla = perm.max() + 1
        bla = np.arange(bla, bla + m).repeat(2)
        perm = np.concatenate((perm, bla), axis=0)

        x = np.array([np.where(perm == i) for i in last_perm]).flatten()
        perms.append(x)

    return perms[::-1]
