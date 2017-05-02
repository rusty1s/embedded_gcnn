from __future__ import division

from six.moves import xrange

import numpy as np
# import numpy_groupies as npg
import scipy.sparse as sp

from .distortion import perm_adj


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


def cut_new(adj, rid=None):
    if rid is None:
        rid = np.random.permutation(np.arange(adj.shape[0]))

    # print(adj.toarray())

    # Sort data by rid
    adj = perm_adj(adj, rid)
    perm = np.argsort(adj.row)
    rows = adj.row[perm]
    cols = adj.col[perm]
    data = adj.data[perm]
    # degree = 1 / npg.aggregate(rows, data, func='sum')
    # data = data * (degree[rows] + degree[cols])
    # print(weights)
    # print(adj.row)
    # print(adj.col)

    n = adj.shape[0]
    cluster_map = np.full(n, -1, np.int32)

    _, rowlength = np.unique(rows, return_counts=True)

    rowend = 0
    cluster_count = 0
    for r in xrange(n):
        # print('r', r)
        rowstart = rowend
        rowend += rowlength[r]

        if cluster_map[r] >= 0:
            continue

        # print('drin')

        cluster_map[r] = cluster_count
        cluster_count += 1

        c = cols[rowstart:rowend]

        cluster = cluster_map[c]
        filter_map = np.where(cluster == -1)[0]
        # print('filter', filter_map)

        if filter_map.size == 0:
            continue

        # print('drin')

        c = c[filter_map]
        d = data[rowstart:rowend]
        d = d[filter_map]
        # print(d)

        min_idx = np.argmin(d)
        cluster_map[c[min_idx]] = cluster_count - 1

    # print('rowstart', rowstart)
    # print('rowlength', rowlength)

    # def _sort(cols):
    #     # Abort early if row already clustered.
    #     # return 0
    #     v = cluster_map[_sort.row]
    #     if v >= 0:
    #         return v

    #     cluster_map[_sort.row] = _sort.row

    #     # Filter cols which are already clustered.
    #     cluster = cluster_map[cols]
    #     filter_map = np.where(cluster >= 0)[0]

    #     # No unclustered adjacent nodes => single node cluster.
    #     if filter_map.size > 0:

    #         w = data[_sort.rowlength:_sort.rowlength+cols.size]
    #         cols = cols[filter_map]
    #         w = w[filter_map]

    #         min_idx = np.argmin(w)
    #         cluster_map[cols[min_idx]] = _sort.row

    #     _sort.row += 1
    #     _sort.rowlength += cols.size
    #     return _sort.row

    # _sort.row = 0
    # _sort.rowlength = 0

    # # Sort groups by minimizing normalized cut.
    # npg.aggregate(adj.row, adj.col, func=_sort)
    # print(a)

    # print(adj.row)

    # print(rows)

    # print(degree)

    return cluster_map
