import numpy as np
import numpy_groupies as npg
import scipy.sparse as sp


def segmentation_adjacency(segmentation):
    # Get centroids.
    idx = np.indices(segmentation.shape)
    ys = npg.aggregate(segmentation.flatten(), idx[0].flatten(), func='mean')
    ys = np.reshape(ys, (-1, 1))
    xs = npg.aggregate(segmentation.flatten(), idx[1].flatten(), func='mean')
    xs = np.reshape(xs, (-1, 1))
    points = np.concatenate((ys, xs), axis=1)

    # Get mass.
    nums, mass = np.unique(segmentation, return_counts=True)
    n = nums.shape[0]

    # Get adjacency (https://goo.gl/y1xFMq).
    # TODO make better
    tmp = np.zeros((n+1, n+1), np.bool)
    a, b = segmentation[:-1, :], segmentation[1:, :]
    tmp[a[a != b], b[a != b]] = True

    a, b = segmentation[:, :-1], segmentation[:, 1:]
    tmp[a[a != b], b[a != b]] = True

    result = tmp | tmp.T
    result = result.astype(np.uint8)

    rowlist = [np.flatnonzero(row) for row in result[:-1]]
    row = np.concatenate(rowlist, axis=0)
    collist = [np.full((rowlist[i].shape[0]), i) for i in range(len(rowlist))]
    col = np.concatenate(collist, axis=0)
    data = np.ones_like(row, dtype=np.uint8)
    adj = sp.coo_matrix((data, (row, col)), (n, n))

    return adj, points, mass
