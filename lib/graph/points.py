from six.moves import xrange

import numpy as np
import scipy.sparse as sp


def points_to_embedded(points, adj, dtype=np.float32):
    """Builds an embedded adjacency matrix based on points of nodes."""

    # Initialize helper variables.
    n = adj.shape[0]
    rows, cols, _ = sp.find(adj)  # Ignore edge weights of `adj`.
    nnz = rows.shape[0]
    dists = np.empty((nnz), dtype=dtype)
    rads = np.empty((nnz), dtype=dtype)

    for i in xrange(nnz):
        # Calculate distance and angle of each edge vector.
        vector = points[cols[i]] - points[rows[i]]
        dists[i] = np.sum(np.power(vector, 2))
        rad = np.arctan2(vector[0], vector[1])
        rads[i] = rad if rad > 0 else rad + 2 * np.pi

    adj_dist = sp.coo_matrix((dists, (rows, cols)), (n, n))
    adj_rad = sp.coo_matrix((rads, (rows, cols)), (n, n))

    return adj_dist, adj_rad
