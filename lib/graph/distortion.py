from six.moves import xrange

import numpy as np
import scipy.sparse as sp


def perm_adj(adj, perm):
    """Permute an adjacency matrix given a permutation. The permutation can be
    greater or smaller than the number of nodes removing nodes or adding fake
    nodes."""

    adj = adj.tocoo(copy=True)

    # Append nodes that should get removed in the process to the end of the
    # permutation.
    nodes_to_remove = np.setdiff1d(np.arange(adj.shape[0]), perm)
    perm_new = np.concatenate((perm, nodes_to_remove), axis=0)

    n = adj.shape[0]
    n_new = perm_new.shape[0]

    # Extend adjacency to contain isolated nodes.
    if n_new > n:
        rows = sp.coo_matrix((n_new - n, n), dtype=adj.dtype)
        cols = sp.coo_matrix((n_new, n_new - n), dtype=adj.dtype)
        adj = sp.vstack([adj, rows])
        adj = sp.hstack([adj, cols])

    sorted_perm = np.argsort(perm_new)
    adj.row = sorted_perm[adj.row]
    adj.col = sorted_perm[adj.col]

    # Slice matrix to final shape.
    n_new = perm.shape[0]
    return adj.tocsr()[0:n_new, :].tocsc()[:, 0:n_new].tocoo()


def perm_features(features, perm):
    """Permute a feature matrix given a permutation. The permutation can be
    greater or smaller than the number of nodes removing nodes or adding fake
    nodes."""

    n, k = features.shape
    n_new = perm.shape[0]

    # Features of none existing nodes should only contain zeros.
    features_new = np.zeros((n_new, k), features.dtype)

    for i in xrange(n_new):
        tid = perm[i]
        features_new[i] = features[tid] if tid < n else 0

    return features_new


def pad_adj(adj, size):
    """Pad an adjacency matrix by appending zeros."""

    return perm_adj(adj, np.arange(size))


def pad_features(features, size):
    """Pad a feature matrix by appending zeros."""

    return perm_features(features, np.arange(size))
