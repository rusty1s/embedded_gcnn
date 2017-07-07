import numpy as np


def node_selection(points, size, stride=1, delta=1):
    # Find coordinate max values.
    y_min = points[:, :1].min()
    x_min = points[:, -1:].min()
    x_max = points[:, -1:].max()
    w_max = x_max - x_min

    # Translate points to min zero.
    points = points - np.array([y_min, x_min])

    # Scale y-coordinates to natural numbers.
    points[:, :1] = w_max * np.floor(points[:, :1] / delta)

    # Sort points.
    points = points.sum(axis=1)
    order = np.argsort(points)

    # Stride and slice points.
    idx = np.arange(np.min([size * stride, order.shape[0]]), step=stride)

    # Fill the rest of the nodes with -1 until we reach the given size.
    filler = -np.ones(np.max([size - idx.shape[0], 0]))
    return np.concatenate([order[idx], filler], axis=0)


def neighborhood_selection(idx, adj_dist, adj_rad, size):
    N, _ = adj_dist.shape
    count = 1

    neighborhood = [idx]

    while neighborhood.count < size or neighborhood.count == N:

        pass

    return neighborhood

    # Was soll passieren?
    # Co
    # Hole die Rows von idx
    pass


def fill_features(receptive_field, features):
    pass
