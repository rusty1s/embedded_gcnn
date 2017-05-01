from ..segmentation import segmentation_adjacency
from ..graph import coarsen_adj, perm_features
from ..tf import sparse_to_tensor


def preprocess_pipeline(image,
                        segmentation_algorithm,
                        feature_extraction_algorithm,
                        levels,
                        scale_invariance=False,
                        stddev=1):

    segmentation = segmentation_algorithm(image)

    adj, points, mass = segmentation_adjacency(segmentation)

    adjs_dist, adjs_rad, perm = coarsen_adj(adj, points, mass, levels,
                                            scale_invariance, stddev)

    features = feature_extraction_algorithm(segmentation, image)
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
