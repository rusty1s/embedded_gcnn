from six.moves import xrange

import numpy as np
import scipy.sparse as sp

from .clustering import normalized_cut
from .distortion import perm_adj


def coarsen_adj(adj, levels, rid=None):
    adjs, parents = _cluster_adj(adj, levels, rid)
    perms = _compute_perms(parents)
    adjs = [perm_adj(adjs[i], perms[i]) for i in xrange(levels + 1)]
    return adjs, perms[0]


def _cluster_adj(adj, levels, rid=None):
    parents = []
    adjs = [adj]

    for _ in xrange(levels):
        # Apply metis.
        cluster_map, rows, cols, weights, _ = normalized_cut(adj, rid)
        parents.append(cluster_map)

        # Compute new coarsened adjacency weights based on cluster map.
        adj = _coarsen_adj_one_level(cluster_map, rows, cols, weights)
        adjs.append(adj)

        # Iterate by degree at next iteration.
        rid = np.argsort(np.array(adj.sum(axis=0)).flatten())

    return adjs, parents


def _coarsen_adj_one_level(cluster_map, rows, cols, weights):
    n_new = cluster_map.max() + 1
    rows = cluster_map[rows]
    cols = cluster_map[cols]
    adj = sp.csr_matrix((weights, (rows, cols)), shape=(n_new, n_new)).tocoo()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def _compute_perms(cluster_maps):
    assert len(cluster_maps) > 0, 'No cluster maps passed.'

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
