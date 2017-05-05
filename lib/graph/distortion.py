import numpy as np
import scipy.sparse as sp


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
    in1d = np.in1d(adj.row, nodes)
    rows = adj.row[in1d]
    cols = adj.col[in1d]
    data = adj.data[in1d]
    in1d = np.in1d(cols, nodes)
    rows = rows[in1d]
    cols = cols[in1d]
    data = data[in1d]
    rows = np.unique(rows, return_inverse=True)[1]
    cols = np.unique(cols, return_inverse=True)[1]
    n = nodes.size
    return sp.coo_matrix((data, (rows, cols)), (n, n))


def filter_features(features, nodes):
    return features[nodes]
