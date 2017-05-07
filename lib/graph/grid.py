import numpy as np
import scipy.sparse as sp

from .distortion import filter_adj


def grid_adj(shape, connectivity=4, dtype=np.float32):
    """Return adjacency matrix of a regular grid."""

    assert connectivity == 4 or connectivity == 8

    h, w = shape

    if connectivity == 4:
        filt = [-w - 2, -1, 1, w + 2]
    else:
        filt = [-w - 3, -w - 2, -w - 1, -1, 1, w + 1, w + 2, w + 3]

    # Build basic rows and cols with +1 padding on all sides.
    n = (h + 1) * (w + 2) - 1
    rows = np.arange(w + 3, n).repeat(connectivity)
    rows = np.reshape(rows, (-1, connectivity))
    cols = rows + filt
    rows = rows.flatten()
    cols = cols.flatten()

    data = np.ones_like(rows, dtype=np.uint8)
    n = (h + 2) * (w + 2)
    adj = sp.coo_matrix((data, (rows, cols)), (n, n))

    # Compute filter nodes.
    rows = np.arange(w + 2, h * (w + 3), w + 2).repeat(w)
    rows = np.reshape(rows, (-1, w))
    cols = np.arange(1, w + 1)
    nodes = (rows + cols).flatten()

    return filter_adj(adj, nodes)


def grid_points(shape, dtype=np.float32):
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    y = y.flatten()
    x = x.flatten()

    z = np.empty((shape[0] * shape[1], 2), dtype)
    z[:, 0] = y
    z[:, 1] = x
    return z


def grid_mass(shape, dtype=np.float32):
    return np.ones(shape[0] * shape[1], dtype)
