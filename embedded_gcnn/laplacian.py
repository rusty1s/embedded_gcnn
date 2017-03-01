from __future__ import division

from six.moves import xrange

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg


def laplacian(A, normalized=False):
    """Return the (normalized) Laplacian of the adjacency matrix."""
    return sp.csgraph.laplacian(A, normalized)


def _lmax(L, normalized=True):
    """Upper-bound on the spectrum."""

    if normalized:
        return 2
    else:
        return sp.linalg.eigsh(L, 1, return_eigenvectors=False)[0]


def _rescale(L, lmax=2):
    """Rescale Laplacian based on upper-bound on the spectrum."""
    return (2 / lmax) * L - sp.eye(L.shape[0])


def chebyshev(L, X, k, normalized=True):
    """Return T_k * X where T_k are the Chebyshev polynomials of order k."""

    L = _rescale(L, _lmax(L, normalized))

    Xt = np.empty((k+1,) + X.shape, L.dtype)

    Xt[0] = X

    if k > 0:
        Xt[1] = L.dot(X)

    for i in xrange(2, k+1):
        Xt[i] = 2 * L.dot(Xt[i-1]) - Xt[i-2]

    return Xt
