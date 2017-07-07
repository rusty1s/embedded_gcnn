import numpy as np


def node_selection(points, size, stride=1, delta=1):
    # Find coordinate max values.
    y_min = points[:, :1].min()
    x_min = points[:, -1:].min()
    x_max = points[:, -1:].max()
    w_max = x_max - x_min

    # Translate points to min zero.
    points = points - np.array([y_min, x_min])

    # Scale y-coordinate to natural number
    points[:, :1] = w_max * np.floor(points[:, :1] / delta)

    # Sort points.
    points = points.sum(axis=1)
    order = np.argsort(points)

    # Stride and slice points.
    idx = np.arange(np.min([size * stride, order.shape[0]]), step=stride)
    return order[idx]
