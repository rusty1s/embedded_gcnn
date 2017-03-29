from __future__ import division

from six.moves import xrange

import sys
import pickle

import numpy as np

from .dataset import DataSet
from .mnist import MNIST
from ..segmentation.algorithm import slic
from ..segmentation.adjacency import segmentation_adjacency
from ..segmentation.feature_extraction import feature_extraction_minimal
from ..graph.embedded_coarsening import coarsen_embedded_adj
from ..graph.distortion import perm_features, perm_adj


class MNISTSlic(DataSet):
    def __init__(self,
                 num_segments=100,
                 compactness=10,
                 max_iterations=10,
                 sigma=0,
                 connectivity=1,
                 locale=False,
                 stddev=1,
                 levels=4,
                 **kwargs):
        if 'data_dir' not in kwargs:
            kwargs['data_dir'] = 'data/mnist/input/slic'

        super(MNISTSlic, self).__init__(**kwargs)

    def _generate_data(self, mnist, num_segments, compactness, max_iterations,
                       sigma, connectivity, locale, stddev, levels):
        adjs_dist = []
        adjs_rad = []
        features = []
        n = np.zeros((levels + 1), np.int32)

        num_examples = mnist.num_train_examples

        for i in xrange(num_examples):
            image = mnist.data.train.images[i]
            image = np.reshape(image, (mnist.height, mnist.width))

            segmentation = slic(image, num_segments, compactness,
                                max_iterations, sigma)

            points, adj, mass = segmentation_adjacency(segmentation,
                                                       connectivity)

            adj_dist, adj_rad, perm = coarsen_embedded_adj(
                points, mass, adj, levels, locale, stddev)
            adjs_dist.append(adj_dist)
            adjs_rad.append(adj_rad)

            feature = feature_extraction_minimal(segmentation, image)
            feature = perm_features(feature, perm)
            features.append(feature)

            n = np.maximum(
                n, np.array([adj_dist[j].shape[0]
                             for j in xrange(levels + 1)]))

            if i % 10 == 0 or i == num_examples - 1:
                sys.stdout.write('\r>> Generating graphs {:.2f}%'.format(
                    100 * (i + 1) / num_examples))
                sys.stdout.flush()

        print()

        # Fill the adjacencies with fake nodes.
        perms = [np.arange(v) for v in n]
        train_data = []
        for i in xrange(num_examples):
            adjs = []
            for j in xrange(levels + 1):
                adj_dist = perm_adj(adjs_dist[i][j], perms[j])
                adj_rad = perm_adj(adjs_rad[i][j], perms[j])
                adjs.append([adj_dist, adj_rad])
            feature = perm_features(features[i], perms[0])
            train_data.append({'features': feature, 'adjacencies': adjs})

            if i % 10 == 0 or i == num_examples - 1:
                sys.stdout.write('\r>> Normalizing graphs {:.2f}%'.format(
                    100 * (i + 1) / num_examples))
                sys.stdout.flush()

        print('\nFinished!')

        pickle.dump(train_data,
                    open('{}/train_data.p'.format(self.data_dir), 'wb'))
        pickle.dump(n, open('{}/info_data.p'.format(self.data_dir), 'wb'))
