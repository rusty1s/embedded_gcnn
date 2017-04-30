from __future__ import division

from six.moves import xrange
import time

import numpy as np

from .clustering import normalized_cut
from .coarsening import compute_perms, _coarsen_clustered_adj
from .distortion import perm_adj
from .adjacency import points_to_adj


def coarsen_embedded_adj(adj,
                         points,
                         mass,
                         levels,
                         scale_invariance=False,
                         stddev=1,
                         rid=None):
    # Coarse graph a defined number of levels deep.
    t = time.process_time()
    adjs_dist, adjs_rad, cluster_maps = _coarsen_embedded_adj(
        adj, points, mass, levels, scale_invariance, stddev, rid)
    print('C1: {:.5f}s'.format(time.process_time() - t))

    # Permutate adjacencies to a binary tree for an efficient O(n) pooling.
    t = time.process_time()
    perms = compute_perms(cluster_maps)
    print('C2: {:.5f}s'.format(time.process_time() - t))
    t = time.process_time()
    adjs_dist = [perm_adj(adjs_dist[i], perms[i]) for i in xrange(levels + 1)]
    adjs_rad = [perm_adj(adjs_rad[i], perms[i]) for i in xrange(levels + 1)]
    print('C3: {:.5f}s'.format(time.process_time() - t))

    return adjs_dist, adjs_rad, perms[0]


def _coarsen_embedded_adj(adj,
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
        adj, points, mass = _coarsen_clustered_embedded_adj(
                                                            adj, points, mass,
                                                            cluster_map)

        # Compute to distance/radian adjacency.
        adj_dist, adj_rad = points_to_adj(adj, points, scale_invariance,
                                          stddev)
        adjs_dist.append(adj_dist)
        adjs_rad.append(adj_rad)

        # Iterate by degree at next iteration.
        rid = np.argsort(np.array(adj_dist.sum(axis=0)).flatten())

    return adjs_dist, adjs_rad, cluster_maps


def _coarsen_clustered_embedded_adj(adj, points, mass, cluster_map):
    n_new = cluster_map.max() + 1

    # Calculate coarsened adjacency matrix.
    adj_new = _coarsen_clustered_adj(cluster_map, adj)
    adj_new = adj_new.astype(np.bool).astype(adj_new.dtype)

    # Calculate new points and masses.
    perm = np.argsort(cluster_map)
    _, rowstart, rowlength = np.unique(
        cluster_map[perm], return_index=True, return_counts=True)

    mass_new = np.empty((n_new), dtype=mass.dtype)
    points_new = np.empty((n_new, 2), dtype=points.dtype)

    for i in xrange(rowstart.shape[0]):
        vi = perm[rowstart[i]]
        if rowlength[i] == 1:
            mass_new[i] = mass[vi]
            points_new[i] = points[vi]
        else:
            vj = perm[rowstart[i] + 1]
            mass_new[i] = mass[vi] + mass[vj]
            points_new[i] = (
                mass[vi] * points[vi] + mass[vj] * points[vj]) / mass_new[i]

    return adj_new, points_new, mass_new
