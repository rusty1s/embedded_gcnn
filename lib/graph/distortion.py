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
    features_new = np.zeros((n_new, k), features.dtype)
    for i in xrange(n_new):
        tid = perm[i]
        if tid < n:
            features_new[i] = features[tid]

    return features_new


def perm_batch_of_features(batch, perm=None):
    if perm is None:
        perm = np.random.permutation(np.arange(batch.shape[1]))

    batch_size, _, k = batch.shape
    n_new = perm.shape[0]

    # Features of none existing nodes should only contain zeros.
    batch_new = np.zeros((batch_size, n_new, k), batch.dtype)
    for i in xrange(batch_size):
        batch_new[i] = perm_features(batch[i], perm)

    return batch_new
