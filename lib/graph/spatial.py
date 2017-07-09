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

    # Fill the rest of the nodes with id -1 until we reach the given size.
    fake = -np.ones(np.max([size - idx.shape[0], 0]), dtype=np.int64)
    return np.concatenate([order[idx], fake], axis=0)


def neighborhood_selection(idx, points, adj, size):
    if idx == -1:
        return np.ones((size), np.int64) * adj.shape[0]

    nodes = np.array([idx])
    current_nodes = np.array([idx])

    while nodes.shape[0] < size and nodes.shape[0] < adj.shape[0]:
        # Calculate all neighbors of current iteration.
        neighbor_idx = np.where(np.in1d(adj.row, current_nodes))[0]
        neighbor_col = adj.col[neighbor_idx]
        neighbor_col = np.unique(neighbor_col)
        filter_idx = np.where(np.in1d(neighbor_col, nodes, invert=True))[0]
        neighbor_col = neighbor_col[filter_idx]

        # Calculate vectors.
        vectors_y = points[idx, 0] - points[neighbor_col, 0]
        vectors_x = points[neighbor_col, 1] - points[idx, 1]
        rads = np.arctan2(vectors_x, vectors_y)
        rads = np.where(rads > 0, rads, rads + 2 * np.pi)

        # Sort by radians.
        order = np.argsort(rads)
        neighbor_col = neighbor_col[order]

        # Append to nodes and iterate over current neighbors in next step.
        nodes = np.concatenate([nodes, neighbor_col], axis=0)
        current_nodes = neighbor_col

    # Append fake nodes.
    nodes = nodes[:size]
    fake = np.ones(np.max([size - nodes.shape[0], 0])) * adj.shape[0]
    return np.concatenate([nodes, fake], axis=0)


def receptive_fields(points,
                     adj,
                     node_size,
                     neighborhood_size,
                     node_stride=1,
                     delta=1):
    nodes = node_selection(points, node_size, node_stride, delta)

    fields = np.zeros((node_size, neighborhood_size), dtype=np.int64)
    for idx, node in enumerate(nodes):
        fields[idx] = neighborhood_selection(node, points, adj,
                                             neighborhood_size)
    return fields


# def fill_features(receptive_fields, features):
#     features = np.concatenate([features, np.zeros_like(features[0])], axis=0)
#     return features[receptive_fields]
