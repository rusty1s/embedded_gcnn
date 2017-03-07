import numpy as np
import scipy.sparse as sp


def perm_adj(adj, perm=None):
    if perm is None:
        perm = np.random.permutation(np.arange(adj.shape[0]))

    n = adj.shape[0]
    n_new = perm.shape[0]

    assert n_new >= n, 'Invalid shapes'

    if n_new >= n:
        rows = sp.coo_matrix((n_new - n, n), dtype=adj.dtype)
        cols = sp.coo_matrix((n_new, n_new - n), dtype=adj.dtype)
        adj = sp.vstack([adj, rows])
        adj = sp.hstack([adj, cols])

    else:
        adj = adj.copy()

    adj.row = np.array(perm)[adj.row]
    adj.col = np.array(perm)[adj.col]
    return adj
