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
