from __future__ import division

import numpy as np
import scipy.sparse as sp


def max_weight(A):
    """Return the maximum weight edge of the adjacency matrix."""
    return A.max()


def normalize(A, max_value):
    """Normalize adjacency matrix to interval [0, max_value]."""
    return (1 / max_value) * A


def gaussian(A, sigma=1):
    """Return gaussian kernel representation of distance adjacency matrix."""
    A = A.copy()
    A.data = np.exp(- A.data ** 2 / (2 * sigma ** 2))
    return A


def grid(width, height, connectivity=4, dtype=np.float32):
    pass
    # x = np.linspace(0, 1, width, dtype=dtype)
    # y = np.linspace(0, 1, height, dtype=dtype)

    # xx, yy = np.meshgrid(x, y)
    # z = np.empty((width * height, 2), dtype)
    # z[:, 0] = xx.reshape(width * height)
    # z[:, 1] = yy.reshape(width * height)

    # idx = np.argsort(d)[:, 1:connectivity+1]

    # d.sort()
    # d = d[:, 1:connectivity+1]
    # return d, idx
