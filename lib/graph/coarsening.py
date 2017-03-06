from six.moves import xrange

import numpy as np
import scipy.sparse as sp


def coarsen_adj(adj, level=1):
    rid = np.random.permutation(np.arange(adj.shape[0]))
    degree = np.array(adj.sum(axis=1)).flatten()

    # Sort by row index.
    rows, cols, vals = sp.find(adj)
    perm = np.argsort(rows)
    rows = rows[perm]
    cols = cols[perm]
    vals = vals[perm]
    # TODO: fix vals so that it represents w * (1/d + 1/d)

    # Get the beginning indices and the count of a new row of rows
    _, rowstart, rowlength = np.unique(
        rows, return_index=True, return_counts=True)

    clusters = np.zeros_like(rowstart, np.int32) - 1
    for i in xrange(rowstart.shape[0]):
        tid = rid[i]  # iterate randomly

        if clusters[tid] >= 0:  # already marked
            continue

        # start = rowstart[tid]
        sliced_vals = vals[rowstart[tid]:rowstart[tid]+rowlength[tid]]
        j = np.argmax(sliced_vals)

        if sliced_vals[j] > 0:
            # Get neighbor index.
            neighbor = cols[rowstart[tid]+j]
            
            # Set cluster.
            clusters[tid] = neighbor
            clusters[neighbor] = tid

            # Set edge weights to zero if the contain one of the nodes.
            vals[np.where(np.logical_or(cols == tid, cols == neighbor))] = 0
        else:
            print(tid)
            # No unmarked neighbor found.
            clusters[tid] = tid

    return clusters

