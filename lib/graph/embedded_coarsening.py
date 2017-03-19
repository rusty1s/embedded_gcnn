from __future__ import division

from six.moves import xrange

import numpy as np

from .clustering import normalized_cut
from .coarsening import compute_perms, _coarsen_clustered_adj
from .distortion import perm_adj
from .embedding import points_to_embedded
from .adjacency import normalize_adj, invert_adj


def coarsen_embedded_adj(points, mass, adj, levels, sigma=1, rid=None):
    # Coarse graph a defined number of levels deep.
    adjs_dist, adjs_rad, cluster_maps = _coarsen_embedded_adj(
        points, mass, adj, levels, sigma, rid)

    # Permutate adjacencies to a binary tree for an efficient O(n) pooling.
    perms = compute_perms(cluster_maps)
    adjs_dist = [perm_adj(adjs_dist[i], perms[i]) for i in xrange(levels + 1)]
    adjs_rad = [perm_adj(adjs_rad[i], perms[i]) for i in xrange(levels + 1)]

    return adjs_dist, adjs_rad, perms[0]


def _coarsen_embedded_adj(points, mass, adj, levels, sigma=1, rid=None):
    adj_dist, adj_rad = points_to_embedded(points, adj)
    adj_dist = invert_adj(normalize_adj(adj_dist), sigma)

    adjs_dist = [adj_dist]
    adjs_rad = [adj_rad]
    cluster_maps = []
    for _ in xrange(levels):
        # Calculate normalized cut clustering.
        cluster_map = normalized_cut(adj_dist, rid)
        cluster_maps.append(cluster_map)

        # Coarsen adjacency.
        points, mass, adj = _coarsen_clustered_embedded_adj(cluster_map,
                                                            points, mass, adj)

        # Compute to distance/radian adjacency.
        adj_dist, adj_rad = points_to_embedded(points, adj)
        adj_dist = invert_adj(normalize_adj(adj_dist), sigma)
        adjs_dist.append(adj_dist)
        adjs_rad.append(adj_rad)

        # Iterate by degree at next iteration.
        rid = np.argsort(np.array(adj_dist.sum(axis=0)).flatten())

    return adjs_dist, adjs_rad, cluster_maps


def _coarsen_clustered_embedded_adj(cluster_map, points, mass, adj):
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

    return points_new, mass_new, adj_new
