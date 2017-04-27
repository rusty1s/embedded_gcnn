from six.moves import xrange

from ..segmentation.adjacency import segmentation_adjacency
from ..graph.embedded_coarsening import coarsen_embedded_adj
from ..graph.distortion import perm_features
from ..tf.convert import sparse_to_tensor


def pipeline(image,
             segmentation_algorithm,
             feature_extraction_algorithm,
             levels,
             locale=False,
             stddev=1,
             connectivity=2):

    segmentation = segmentation_algorithm(image)

    points, adj, mass = segmentation_adjacency(segmentation, connectivity)

    adjs_dist, adjs_rad, perm = coarsen_embedded_adj(points, mass, adj, levels,
                                                     locale, stddev)

    features = feature_extraction_algorithm(segmentation, image)
    features = perm_features(features, perm)

    adjs_dist = [sparse_to_tensor(A) for A in adjs_dist]
    adjs_rad = [sparse_to_tensor(A) for A in adjs_rad]

    return features, adjs_dist, adjs_rad


def batch_pipeline(images,
                   segmentation_algorithm,
                   feature_extraction_algorithm,
                   levels,
                   locale=False,
                   stddev=1,
                   connectivity=2):

    features = []
    adjs_dist = []
    adjs_rad = []

    for i in xrange(images.shape[0]):
        f, dist, rad = pipeline(images[i], segmentation_algorithm,
                                feature_extraction_algorithm, levels, locale,
                                stddev, connectivity)
        features.append(f)
        adjs_dist.append(dist)
        adjs_rad.append(rad)

    return features, adjs_dist, adjs_rad
