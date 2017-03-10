from six.moves import xrange

import numpy as np
import scipy.sparse as sp


def perm_adj(adj, perm=None):
    if perm is None:
        perm = np.random.permutation(np.arange(adj.shape[0]))

    n = adj.shape[0]
    n_new = perm.shape[0]
    assert n_new >= n, 'Invalid shapes'

    adj = adj.tocoo(copy=True)

    # Extend adjacency to contain isolated nodes.
    if n_new > n:
        rows = sp.coo_matrix((n_new - n, n), dtype=adj.dtype)
        cols = sp.coo_matrix((n_new, n_new - n), dtype=adj.dtype)
        adj = sp.vstack([adj, rows])
        adj = sp.hstack([adj, cols])

    adj.row = np.array(perm)[adj.row]
    adj.col = np.array(perm)[adj.col]
    return adj


def perm_features(features, perm=None):
    if perm is None:
        perm = np.random.permutation(np.arange(features.shape[0]))

    n, k = features.shape
    n_new = perm.shape[0]
    assert n_new >= n, 'Invalid shapes'

    # Features of none existing nodes should only contain zeros.
    new_features = np.zeros(n_new, k)
    for i in xrange(n_new):
        tid = perm[i]
        if tid < n:
            new_features[i] = new_features[tid]

    return new_features
