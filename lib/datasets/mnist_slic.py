from six.moves import xrange

import pickle
import time

import numpy as np

from .dataset import DataSet
from .mnist import MNIST
from ..segmentation.algorithm import slic
from ..segmentation.adjacency import segmentation_adjacency
from ..segmentation.feature_extraction import feature_extraction_minimal
from ..graph.embedded_coarsening import coarsen_embedded_adj
from ..graph.embedding import partition_embedded_adj
from ..graph.distortion import perm_features


class MNISTSlic(DataSet):
    def __init__(self,
                 num_segments=100,
                 compactness=10,
                 max_iterations=10,
                 sigma=0,
                 levels=3,
                 **kwargs):
        if 'data_dir' not in kwargs:
            kwargs['data_dir'] = 'data/mnist/input/slic'

        super(MNISTSlic, self).__init__(**kwargs)

        mnist = MNIST()

        train_data = []

        n_max_1 = 0
        n_max_2 = 0
        n_max_3 = 0
        n_max_4 = 0
        num_examples = 20

        t_seg = 0
        t_adj = 0
        t_feature = 0
        t_coarsen = 0
        t_partition = 0
        for i in xrange(num_examples):
            image = mnist.data.train.images[i]
            image = np.reshape(image, (mnist.height, mnist.width))

            t_start = time.process_time()
            segmentation = slic(image, num_segments, compactness,
                                max_iterations, sigma)
            t_seg += time.process_time() - t_start
            t_start = time.process_time()

            points, adj, mass = segmentation_adjacency(
                segmentation, connectivity=1)
            t_adj += time.process_time() - t_start
            t_start = time.process_time()
            features = feature_extraction_minimal(segmentation, image)
            t_feature += time.process_time() - t_start
            t_start = time.process_time()

            adjs_dist, adjs_rad, perm = coarsen_embedded_adj(
                points, mass, adj, levels, locale=False, sigma=1)
            t_coarsen += time.process_time() - t_start
            t_start = time.process_time()

            # n_max_1 = max(adjs_dist[0].shape[0], n_max_1)
            # n_max_2 = max(adjs_dist[1].shape[0], n_max_2)
            # n_max_3 = max(adjs_dist[2].shape[0], n_max_3)
            # n_max_4 = max(adjs_dist[3].shape[0], n_max_4)

            adjs_1 = partition_embedded_adj(
                adjs_dist[0],
                adjs_rad[0],
                num_partitions=8,
                offset=0.125 * np.pi)
            adjs_2 = partition_embedded_adj(
                adjs_dist[1],
                adjs_rad[1],
                num_partitions=8,
                offset=0.125 * np.pi)
            adjs_3 = partition_embedded_adj(
                adjs_dist[2],
                adjs_rad[2],
                num_partitions=8,
                offset=0.125 * np.pi)
            adjs_4 = partition_embedded_adj(
                adjs_dist[3],
                adjs_rad[3],
                num_partitions=8,
                offset=0.125 * np.pi)
            adjs = [adjs_1, adjs_2, adjs_3, adjs_4]
            t_partition += time.process_time() - t_start
            t_start = time.process_time()

            # features = perm_features(features, perm)
            # train_data.append({'adjacencies': adjs, 'features': features})
            print(100 * (i+1) / num_examples)

        print(t_seg)
        print(t_adj)
        print(t_feature)
        print(t_coarsen)
        print(t_partition)
        print('complete time', t_seg + t_adj + t_feature + t_coarsen + t_partition)
        print(n_max_1)
        print(n_max_2)
        print(n_max_3)
        print(n_max_4)

        # pickle.dump(train_data,
        #             open('{}/train_data.p'.format(self.data_dir), 'wb'))
        # pickle.dump(mnist.data.train.labels[:200],
        #             open('{}/train_labels.p'.format(self.data_dir), 'wb'))
        # pickle.dump(n_max, open('{}/n_max.p'.format(self.data_dir), 'wb'))
