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
    # Coarse graph a defined number of levels deep.
    adjs_dist, adjs_rad, cluster_maps = _coarsen_adj(
        adj, points, mass, levels, scale_invariance, stddev, rid)

    # Permutate adjacencies to a binary tree for an efficient O(n) pooling.
    perms = compute_perms(cluster_maps)
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

    points_y = mass * points[:, :1].flatten()
    points_y = npg.aggregate(cluster_map, points_y, func='mean')

    points_x = mass * points[:, 1:].flatten()
    points_x = npg.aggregate(cluster_map, points_x, func='mean')

    mass = npg.aggregate(cluster_map, mass, func='sum')

    points_y = points_y / mass
    points_x = points_x / mass

    points_y = np.reshape(points_y, (-1, 1))
    points_x = np.reshape(points_x, (-1, 1))
    points = np.concatenate((points_y, points_x), axis=1)

    return adj, points, mass


def compute_perms(cluster_maps):
    # Last permutation is the ordered list of the number of clusters in the
    # last cluster map.
    perms = [np.arange(np.max(cluster_maps[-1] + 1))]

    # Iterate backwards through cluster_maps.
    for i in reversed(xrange(len(cluster_maps))):
        cluster_map = cluster_maps[i]

        cur_singleton = len(cluster_map)

        perm = []
        for j in perms[-1]:
            # Indices of the cluster map that correspond to the calculated
            # permutation.
            perm_nodes = list(np.where(cluster_map == j)[0])

            # Add fake nodes if neccassary.
            if len(perm_nodes) == 1:
                perm_nodes.append(cur_singleton)
                cur_singleton += 1
            elif len(perm_nodes) == 0:
                perm_nodes.append(cur_singleton)
                perm_nodes.append(cur_singleton + 1)
                cur_singleton += 2

            # Append the two nodes to the permutation.
            perm.extend(perm_nodes)

        perms.append(np.array(perm))

    # Reverse permutations.
    return perms[::-1]
