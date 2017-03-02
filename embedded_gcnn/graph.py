from __future__ import division

from six.moves import xrange

import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as sd


def normalize(A):
    """Normalize adjacency matrix to interval [0, 1]."""
    return (1 / A.max()) * A


def gaussian(A, sigma=1):
    """Return (inverted) gaussian kernel representation of adjacency matrix.
    Note that this methods only accepts squared distance adjacency matrices."""

    if sp.issparse(A):
        A = A.copy()
        A.data = np.exp(- A.data / (2 * sigma * sigma))
        return A
    else:
        return np.exp(- A / (2 * sigma * sigma))


def grid(shape, connectivity=4, dtype=np.float32):
    assert connectivity == 4 or connectivity == 8

    height, width = shape
    num_nodes = height * width
    A = sp.lil_matrix((num_nodes, num_nodes), dtype=dtype)

    for i in xrange(0, num_nodes):
        if i % width > 0:  # left node
            A[i, i-1] = 1
        if i % width < width - 1:  # right node
            A[i, i+1] = 1
        if i >= width:  # top node
            A[i, i-width] = 1
            if connectivity > 4 and i % width > 0:
                A[i, i-width-1] = 2
            if connectivity > 4 and i % width < width - 1:
                A[i, i-width+1] = 2
        if i < height * width - width:  # bottom node
            A[i, i+width] = 1
            if connectivity > 4 and i % width > 0:
                A[i, i+width-1] = 2
            if connectivity > 4 and i % width < width - 1:
                A[i, i+width+1] = 2

    return A
