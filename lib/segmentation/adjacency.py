import numpy as np
import numpy_groupies as npg
import scipy.sparse as sp


def segmentation_adjacency(segmentation):
    # Get centroids.
    idx = np.indices(segmentation.shape)
    ys = npg.aggregate(segmentation.flatten(), idx[0].flatten(), func='mean')
    xs = npg.aggregate(segmentation.flatten(), idx[1].flatten(), func='mean')
    ys = np.reshape(ys, (-1, 1))
    xs = np.reshape(xs, (-1, 1))
    points = np.concatenate((ys, xs), axis=1)

    # Get mass.
    nums, mass = np.unique(segmentation, return_counts=True)
    n = nums.shape[0]

    # Get adjacency (https://goo.gl/y1xFMq).
    tmp = np.zeros((n, n), np.bool)
    a, b = segmentation[:-1, :], segmentation[1:, :]
    tmp[a[a != b], b[a != b]] = True

    a, b = segmentation[:, :-1], segmentation[:, 1:]
    tmp[a[a != b], b[a != b]] = True

    result = tmp | tmp.T
    result = result.astype(np.uint8)
    adj = sp.coo_matrix(result)

    return adj, points, mass
