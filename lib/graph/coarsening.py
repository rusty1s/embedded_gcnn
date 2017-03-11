from __future__ import division

from six.moves import xrange

import numpy as np
import scipy.sparse as sp


def _cluster_adj(adj, rid=None):
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

    # Generate random iteration permutation.
    if rid is None:
        rid = np.random.permutation(np.arange(adj.shape[0]))

    assert adj.shape[0] == rid.shape[0]

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

            cluster_map[tid] = cur_id
            cluster_map[neighbor] = cur_id

            # Set edge weights to zero if they contain one of the nodes.
            weights[np.where(np.logical_or(cols == tid, cols == neighbor))] = 0
        else:
            cluster_map[tid] = cur_id

            # Set edge weights to zero if they contain one of the nodes.
            weights[np.where(cols == tid)] = 0

        # print(cur_id)
        cur_id += 1

    return cluster_map
