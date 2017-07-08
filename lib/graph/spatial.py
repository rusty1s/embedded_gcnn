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
    filler = -np.ones(np.max([size - idx.shape[0], 0]))
    return np.concatenate([order[idx], filler], axis=0)


def neighborhood_selection(idx, points, adj, size):
    # neighborhood = np.array([idx], np.int32)

    # current_neighbors = []
    # new_nodes = []
    nodes = np.array([idx])
    current_nodes = np.array([idx])

    while nodes.shape[0] < size or nodes.shape[0] < adj.shape[0]:
        # Calculate all neighbors.
        neighbor_idx = np.where(np.in1d(adj.row, current_nodes))[0]
        neighbor_col = adj.col[neighbor_idx]
        neighbor_col = np.unique(neighbor_col)
        # TODO: Filter nodes that are already calculated

        # Calculate vectors.
        vectors = points[neighbor_col] - points[idx]
        rads = np.arctan2(vectors[:, 1], vectors[:, 0])
        rads = np.where(rads > 0, rads, rads + 2 * np.pi)

        # Sort by radians.
        order = np.argsort(rads)
        neighbor_col = neighbor_col[order]

        # Append to nodes and iterate over current neighbors in next step.
        nodes = np.concatenate([nodes, neighbor_col], axis=0)
        current_nodes = neighbor_col

    return nodes[:size]


# def build_receptive_fields(points,
#                            adj_dist,
#                            adj_rad,
#                            node_size,
#                            neighborhood_size,
#                            delta=1):

#     nodes = node_selection(points)
#     neighborhoods = []
#     for node in nodes:
#         neighborhoods.append(
#             neighborhood_selection(node, adj_dist, adj_rad, neighborhood_size))
#     return np.array(neighborhoods)

# def fill_features(receptive_field, features):
#     features = np.concatenate((features, np.zeros_like(features[0])))
#     features[receptive_field]
#     pass
