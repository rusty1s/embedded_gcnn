from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg


def laplacian(A, normalized=True):
    """Return the Laplacian of the adjacency matrix."""

    d = np.array(A.sum(1)).flatten()

    if not normalized:
        D = sp.diags(d)
        return D - A
    else:
        with np.errstate(divide='ignore'):
            d = np.power(d, -0.5)
            d[np.isinf(d)] = 0
        D = sp.diags(d)
        return sp.eye(A.shape[0]) - D.dot(A).transpose().dot(D)


def lmax(L, normalized=True):
    """Upper-bound on the spectrum."""

    if normalized:
        return 2
    else:
        return sp.linalg.eigsh(L, 1, return_eigenvectors=False)[0]


def rescale(L, lmax=2):
    return (2 / lmax) * L - sp.eye(L.shape[0])


def chebyshev(L, X, k, normalized=True):
    """Return T_k * X where T_k are the Chebyshev polynomials of order k."""

    L = rescale(L, lmax(L, normalized))

    Xt = np.empty((k+1, X.shape[0], X.shape[1]), L.dtype)

    Xt[0] = X
    Xt[1] = L.dot(X)

    for i in xrange(2, k+1):
        Xt[k] = 2 * L.dot(Xt[i-1]) - Xt[i-2]

    return Xt[k]


    largest_eigval, _ = sp.linalg.eigsh(laplacian, 1)
    largest_eigval = largest_eigval[0]
    scaled_laplacian = (2 / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])

    t_k = []
    t_k.append(sp.eye(laplacian.shape[0]))  # T_0
    t_k.append(scaled_laplacian)            # T_1

    for _ in xrange(1, k):
        t_k.append(2 * scaled_laplacian.dot(t_k[-1]) - t_k[-2])  # T_i

    return sparse_to_tuple(t_k)


adj = np.array([[0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0]])
adj = sp.coo_matrix(adj)
laplacian = laplacian(adj, normalized=False)
print(chebyshev(laplacian, 3))

