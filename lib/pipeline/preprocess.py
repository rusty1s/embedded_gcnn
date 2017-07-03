from ..segmentation import segmentation_adjacency
from ..graph import coarsen_adj, perm_features


def preprocess_pipeline(image,
                        segmentation_algorithm,
                        feature_extraction_algorithm,
                        levels,
                        connectivity=8,
                        scale_invariance=False,
                        stddev=1):

    segmentation = segmentation_algorithm(image)
    adj, points, mass = segmentation_adjacency(segmentation, connectivity)
    features = feature_extraction_algorithm(segmentation, image)

    adjs_dist, adjs_rad, perm = coarsen_adj(adj, points, mass, levels,
                                            scale_invariance, stddev)

    features = perm_features(features, perm)

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
