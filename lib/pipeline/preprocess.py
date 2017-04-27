from ..segmentation.adjacency import segmentation_adjacency
from ..graph.embedded_coarsening import coarsen_embedded_adj
from ..graph.distortion import perm_features


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

    return adjs_dist, adjs_rad, features
