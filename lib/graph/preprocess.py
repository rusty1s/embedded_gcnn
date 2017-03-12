import numpy as np
import scipy.sparse as sp


def preprocess_adj(adj):
    # Set diagonal entries to one, if the corresponding node has at least one
    # edge.
    degree = np.array(adj.sum(1)).flatten()
    ones = degree.astype(bool).astype(adj.dtype)
    adj = adj.copy()
    adj.setdiag(ones)

    # Calculate normalization.
    degree = degree + ones
    with np.errstate(divide='ignore'):
        degree = np.power(degree, -0.5)
        degree[np.isinf(degree)] = 0
    degree = sp.diags(degree)

    return degree.dot(adj).dot(degree)
