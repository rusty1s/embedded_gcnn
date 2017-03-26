from __future__ import print_function
from __future__ import division

import time
from six.moves import xrange

import numpy as np
import tensorflow as tf

from lib.dataset.mnist import MNIST
from lib.segmentation.algorithm import slic
from lib.segmentation.adjacency import segmentation_adjacency
from lib.segmentation.feature_extraction import feature_extraction_minimal
from lib.graph.embedding import partition_embedded_adj
from lib.graph.embedded_coarsening import coarsen_embedded_adj
from lib.graph.preprocess import preprocess_adj
from lib.graph.sparse import sparse_to_tensor
from lib.graph.distortion import perm_features

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('graph_connectivity', 2, 'TODO')
flags.DEFINE_integer('slic_num_segments', 100, 'TODO')
flags.DEFINE_float('slic_compactness', 10, 'TODO')
flags.DEFINE_integer('slic_max_iterations', 10, 'TODO')
flags.DEFINE_float('slic_sigma', 0, 'TODO')
flags.DEFINE_integer('num_partitions', 8, 'TODO')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 64, 'How many inputs to process at once.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('data_dir', 'data/mnist/input',
                    'Directory for storing input data.')
flags.DEFINE_string('log_dir', 'data/mnist/summaries/conv2d',
                    'Summaries log directory.')
flags.DEFINE_integer('display_step', 10,
                     'How many steps to print logging after.')

data = MNIST(data_dir=FLAGS.data_dir)


def preprocess_image(image):
    segmentation = slic(image, FLAGS.slic_num_segments, FLAGS.slic_compactness,
                        FLAGS.slic_max_iterations, FLAGS.slic_sigma)
    points, adj, mass = segmentation_adjacency(segmentation,
                                               FLAGS.graph_connectivity)
    features = feature_extraction_minimal(segmentation, image)

    adjs_dist, adjs_rad, perm = coarsen_embedded_adj(
        points, mass, adj, levels=4)

    adjs_1 = partition_embedded_adj(
        adjs_dist[0],
        adjs_rad[0],
        num_partitions=FLAGS.num_partitions,
        offset=0.125 * np.pi)
    adjs_2 = partition_embedded_adj(
        adjs_dist[2],
        adjs_rad[2],
        num_partitions=FLAGS.num_partitions,
        offset=0.125 * np.pi)

    adjs_1 = [sparse_to_tensor(preprocess_adj(a)) for a in adjs_1]
    adjs_2 = [sparse_to_tensor(preprocess_adj(a)) for a in adjs_2]

    return adjs_1, adjs_2, perm_features(features, perm)


placeholders = {
    'features': tf.placeholder(tf.float32, name='features'),
    'adjacency_1': [
        tf.sparse_placeholder(tf.float32, name='adjacency_1_{}'.format(i + 1))
        for i in xrange(FLAGS.num_partitions)
    ],
    'adjacency_2': [
        tf.sparse_placeholder(tf.float32, name='adjacency_2_{}'.format(i + 1))
        for i in xrange(FLAGS.num_partitions)
    ],
    'labels': tf.placeholder(tf.int32, [FLAGS.batch_size], 'labels'),
    'dropout': tf.placeholder(tf.float32, [], 'dropout'),
}

for step in xrange(0, 1):
    print('step', step)

    train_images, train_labels = data.next_train_batch(FLAGS.batch_size)
    adjs_1, adjs_2, features = preprocess_image(train_images[0])
    print(features.shape)
