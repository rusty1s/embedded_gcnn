from __future__ import division

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def laplacian(adj, normalized=False):
    """Return the (normalized) laplacian of the adjacency matrix."""
    return sp.csgraph.laplacian(adj, normalized)


def lmax(lap, normalized=True):
    """Upper-bound on the spectrum."""

    if normalized:
        return 2
    else:
        return eigsh(lap, 1, return_eigenvectors=False)[0]


def rescale_lap(lap, lmax=2):
    """Rescale laplacian based on upper-bound on the spectrum."""
    return (2 / lmax) * lap - sp.eye(lap.shape[0])
