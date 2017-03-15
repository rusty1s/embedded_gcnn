from __future__ import division

from six.moves import xrange

import numpy as np
import scipy.sparse as sp

from .distortion import perm_adj


def coarsen_adj(adj, levels, rid=None):
    assert levels > 0, 'Levels must be greater than zero.'

    # Generate levels + 1 graphs.
    adjs = [adj]
    cluster_maps = []
    for i in xrange(levels):
        adj, cluster_map = _coarsen_adj(adjs[-1]) if i > 0 else _coarsen_adj(
            adjs[-1], rid)
        adjs.append(adj)
        cluster_maps.append(cluster_map)

    # Sort adjacencies so we can perform an efficient pooling operation.
    perms = _compute_perms(cluster_maps)
    adjs = [perm_adj(adjs[i], perms[i]) for i in xrange(len(adjs))]

    # Return all adjacencies and first permutation to sort node features.
    return adjs, perms[0]


def _coarsen_adj(adj, rid=None):
    cluster_map = _cluster_adj(adj, rid)

    # Compute new edge weights
    rows, cols, weights = sp.find(adj)
    perm = np.argsort(rows)
    rows = rows[perm]
    cols = cols[perm]
    weights = weights[perm]
    n_new = np.max(cluster_map) + 1
    rows_new = cluster_map[rows]
    cols_new = cluster_map[cols]
    weights_new = weights

    adj_new = sp.csr_matrix(
        (weights_new, (rows_new, cols_new)), shape=(n_new, n_new)).tocoo()
    adj_new.setdiag(0)
    adj_new.eliminate_zeros()
    return adj_new, cluster_map


def _cluster_adj(adj, rid=None):
    # Generate random iteration permutation if neccassary.
    if rid is None:
        rid = np.random.permutation(np.arange(adj.shape[0]))

    assert adj.shape[0] == rid.shape[0], 'Invalid shapes.'

    # Sort by row index.
    rows, cols, weights = sp.find(adj)
    perm = np.argsort(rows)
    rows = rows[perm]
    cols = cols[perm]
    weights = weights[perm]

    # Local normalized cut.
    degree = np.array(adj.sum(axis=1)).flatten()
    weights = weights / degree[rows] + weights / degree[cols]

    # Get the beginning indices and the count of every row.
    _, rowstart, rowlength = np.unique(
        rows, return_index=True, return_counts=True)

    # Initialize empty cluster mapping.
    cluster_map = np.zeros(adj.shape[0], dtype=np.int32) - 1

    cur_id = 0
    for i in xrange(rowstart.shape[0]):
        # Iterate randomly.
        tid = rid[i]

        if cluster_map[tid] >= 0:
            # Node already marked. Skip it.
            continue

        start = rowstart[tid]
        end = start + rowlength[tid]
        weights_i = weights[start:end]

        # Randomly sort sliced weights, so the ordering of nodes is unimportant
        # if two nodes share the same weight.
        perm = np.random.permutation(np.arange(weights_i.shape[0]))
        weights_i = weights_i[perm]

        j = np.argmax(weights_i)

        if weights_i[j] > 0:
            # We found a neighbor.
            cols_i = cols[start:end]
            cols_i = cols_i[perm]
            neighbor = cols_i[j]

            # Add to cluter map.
            cluster_map[tid] = cur_id
            cluster_map[neighbor] = cur_id

            # Set edge weights to zero if they contain one of the nodes.
            weights[np.where(np.logical_or(cols == tid, cols == neighbor))] = 0
        else:
            cluster_map[tid] = cur_id

            # Set edge weights to zero if they contain the node.
            weights[np.where(cols == tid)] = 0

        cur_id += 1

    return cluster_map


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
