import numpy as np


def grid_adj(shape, connectivity=4, dtype=np.float32):
    """Return adjacency matrix of a regular grid."""

    assert connectivity == 4 or connectivity == 8

    # TODO
    raise NotImplementedError


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
