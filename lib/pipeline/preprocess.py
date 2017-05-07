from ..segmentation import segmentation_adjacency
from ..graph import coarsen_adj, perm_features
from ..graph import filter_adj, filter_features
from ..tf import sparse_to_tensor


def preprocess_pipeline(image,
                        segmentation_algorithm,
                        feature_extraction_algorithm,
                        levels,
                        filter_algorithm=None,
                        scale_invariance=False,
                        stddev=1):

    segmentation = segmentation_algorithm(image)
    adj, points, mass = segmentation_adjacency(segmentation)
    features = feature_extraction_algorithm(segmentation, image)

    if filter_algorithm is not None:
        nodes = filter_algorithm(adj, features)
        adj = filter_adj(adj)
        points = filter_features(points)
        mass = filter_features(mass)
        features = filter_features(features)

    adjs_dist, adjs_rad, perm = coarsen_adj(adj, points, mass, levels,
                                            scale_invariance, stddev)

    features = perm_features(features, perm)
    adjs_dist = [sparse_to_tensor(A) for A in adjs_dist]
    adjs_rad = [sparse_to_tensor(A) for A in adjs_rad]

    return features, adjs_dist, adjs_rad


def preprocess_pipeline_fixed(segmentation_algorithm,
                              feature_extraction_algorithm,
                              levels,
                              scale_invariance=False,
                              stddev=1):
    def _preprocess(image):
        return preprocess_pipeline(image, segmentation_algorithm,
                                   feature_extraction_algorithm, levels,
                                   scale_invariance, stddev)

    return _preprocess
