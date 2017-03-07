from __future__ import division

from six.moves import xrange

import numpy as np
import scipy.sparse as sp


def coarsen(adj, levels=1):
    pass


def coarsen_one_level(adj):
    # clusters = cluster_adj(adj)

    return adj


def cluster_adj(adj, rid=None):
    # Sort by row index.
    rows, cols, weights = sp.find(adj)
    perm = np.argsort(rows)
    rows = rows[perm]
    cols = cols[perm]
    weights = weights[perm]

    # Local normalized cut.
    degree = np.array(adj.sum(axis=1)).flatten()
    weights = weights / degree[rows] + weights / degree[cols]

    # Get the beginning indices and the count of a new row of rows
    _, rowstart, rowlength = np.unique(
        rows, return_index=True, return_counts=True)

    # Save nearest neighbor to all nodes.
    clusters = np.zeros_like(rowstart, np.int32) - 1

    if rid is None:
        rid = np.random.permutation(np.arange(adj.shape[0]))

    for i in xrange(rowstart.shape[0]):
        # Iterate randomly.
        tid = rid[i]

        if clusters[tid] >= 0:
            # Node already marked. Skip it.
            continue

        # Get the weights of the row.
        start = rowstart[tid]
        end = start + rowlength[tid]
        weights_i = weights[start:end]

        # Randomly sort sliced weights.
        perm = np.random.permutation(np.arange(weights_i.shape[0]))
        weights_i = weights_i[perm]

        # Find best neighbor.
        j = np.argmax(weights_i)

        if weights_i[j] > 0:
            # We found a neighbor, get its index.
            cols_i = cols[start:end]
            cols_i = cols_i[perm]
            neighbor = cols_i[j]

            # Set cluster.
            clusters[tid] = neighbor
            clusters[neighbor] = tid

            # Set edge weights to zero if they contain one of the nodes.
            weights[np.where(np.logical_or(cols == tid, cols == neighbor))] = 0
        else:
            # No unmarked neighbor found.
            clusters[tid] = tid

    return clusters
