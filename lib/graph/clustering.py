from __future__ import division

from six.moves import xrange

import numpy as np
# import numpy_groupies as npg
import scipy.sparse as sp


def normalized_cut(adj, rid=None):
    if rid is None:
        rid = np.random.permutation(np.arange(adj.shape[0]))

    n = adj.shape[0]
    cluster_map = np.zeros(n, np.int32) - 1
    clustercount = 0

    # Sort by row index.
    rows, cols, weights = sp.find(adj)
    perm = np.argsort(rows)
    rows = rows[perm]
    cols = cols[perm]
    weights = weights[perm]
    degree = np.array(adj.sum(axis=0)).flatten()

    # Get the beginning indices and the count of every row.
    _, rowstart, rowlength = np.unique(
        rows, return_index=True, return_counts=True)

    if rowstart.size == 0:
        return cluster_map + 1

    for r in xrange(n):
        # Iterate randomly.
        tid = rid[r]

        if cluster_map[tid] == -1:  # Not already marked
            cluster_map[tid] = clustercount
            wmax = 0.0
            rs = rowstart[tid]
            bestneighbor = -1

            # Find best neighbor (Normalized Cut).
            for c in range(rowlength[tid]):
                nid = cols[rs + c]
                w = weights[rs + c] * (1.0 / degree[tid] + 1.0 / degree[nid]
                                       ) if cluster_map[nid] == -1 else 0.0
                if w > wmax:
                    wmax = w
                    bestneighbor = nid

            if bestneighbor > -1:
                cluster_map[bestneighbor] = clustercount
            clustercount += 1

    return cluster_map


# import numpy_groupies as npg

# def cut_new(adj, rid=None):
#     if rid is None:
#         rid = np.random.permutation(np.arange(adj.shape[0]))

#     cluster_map = None

#     adj = perm_adj(adj, rid)

#     rows = adj.row
#     cols = adj.col
#     data = adj.data
#     degree = 1 / npg.aggregate(rows, data, func='sum')
#     data = data * (degree[rows] + degree[cols])

#     # SO, jetz ist row in rid order und data can be sorted
#     print(rows)
#     print(cols)
#     print(data)
#     # Data has now normalized cut comparison

#     # Problem
#     # wir wollen jetz rows durchlaufen (in rid order)
#     # und sortieren col dabei???

#     # dann durchlaufen wir row+col und setzen unsere cluster map

#     return cluster_map
