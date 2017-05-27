import numpy as np
import scipy.sparse as sp
import numpy_groupies as npg


def perm_adj(adj, perm):
    """Permute an adjacency matrix given a permutation. The permutation must
    contain all existing nodes in an arbitrary order and can contain fake
    nodes."""

    n = perm.shape[0]
    sorted_perm = np.argsort(perm)
    row = sorted_perm[adj.row]
    col = sorted_perm[adj.col]
    return sp.coo_matrix((adj.data, (row, col)), (n, n))


def perm_features(features, perm):
    """Permute an feature matrix given a permutation. The permutation must
    contain all existing nodes in an arbitrary order and can contain fake
    nodes."""

    n, k = features.shape
    num_fake_nodes = perm.shape[0] - n
    zeros = np.zeros((num_fake_nodes, k), features.dtype)
    features = np.concatenate((features, zeros), axis=0)
    return features[perm]


def filter_adj(adj, nodes):
    # Filter by rows.
    in1d = np.in1d(adj.row, nodes)
    rows = adj.row[in1d]
    cols = adj.col[in1d]
    data = adj.data[in1d]

    # Filter by cols.
    in1d = np.in1d(cols, nodes)
    rows = rows[in1d]
    cols = cols[in1d]
    data = data[in1d]

    # Remap indices to new range.
    rows = np.unique(rows, return_inverse=True)[1]
    cols = np.unique(cols, return_inverse=True)[1]

    n = nodes.size
    return sp.coo_matrix((data, (rows, cols)), (n, n))


def filter_features(features, nodes):
    return features[nodes]


def gray_color_threshold(adj, features, k):
    gray = features[:, :1]
    return np.where(gray >= k)[0]


def gray_color_threshold_fixed(k):
    def _threshold(adj, features):
        return gray_color_threshold(adj, features, k)

    return _threshold


def degree_threshold(adj, features, k):
    # Adjacency must contain one in every entry.
    degree = npg.aggregate(adj.row, adj.data, func='sum')
    return np.where(degree <= k)[0]


def degree_threshold_fixed(k):
    def _threshold(adj, features):
        return degree_threshold(adj, features, k)

    return _threshold
