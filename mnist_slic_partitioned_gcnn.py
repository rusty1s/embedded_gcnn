from __future__ import print_function
from __future__ import division

from six.moves import xrange

import numpy as np
import tensorflow as tf

from lib.dataset.mnist import MNIST
from lib.segmentation.algorithm import slic
from lib.segmentation.adjacency import segmentation_adjacency
from lib.segmentation.feature_extraction import (feature_extraction_minimal,
                                                 NUM_FEATURES_MINIMAL)
from lib.graph.embedding import partition_embedded_adj
from lib.graph.embedded_coarsening import coarsen_embedded_adj
from lib.graph.preprocess import preprocess_adj
from lib.graph.sparse import sparse_to_tensor
from lib.graph.distortion import perm_features
from lib.model.model import Model
from lib.layer.partitioned_gcnn import PartitionedGCNN as Conv
from lib.layer.max_pool_gcnn import MaxPoolGCNN as MaxPool
from lib.layer.fixed_mean_pool import FixedMeanPool as FixedPool
from lib.layer.fc import FC

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('locale_normalization', False,
                     '''Whether to normalize each adjacency locally.''')
flags.DEFINE_integer('graph_connectivity', 2,
                     '''The connectivity between pixels in the segmentation. A
                     connectivity of 1 corresponds to immediate neighbors up,
                     down, left and right, while a connectivity of 2 also
                     includes diagonal neighbors.''')
flags.DEFINE_integer('slic_num_segments', 100, 'TODO')
flags.DEFINE_float('slic_compactness', 10, 'TODO')
flags.DEFINE_integer('slic_max_iterations', 10, 'TODO')
flags.DEFINE_float('slic_sigma', 0, 'TODO')
flags.DEFINE_integer('num_partitions', 8,
                     '''The number of partitions of each graph corresponding to
                     the number of weights for each convolution.''')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 8, 'How many inputs to process at once.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('data_dir', 'data/mnist/input',
                    'Directory for storing input data.')
flags.DEFINE_string('log_dir', 'data/mnist/summaries/slic_partitioned_gcnn',
                    'Summaries log directory.')
flags.DEFINE_integer('display_step', 1,
                     'How many steps to print logging after.')

data = MNIST(data_dir=FLAGS.data_dir)


def preprocess_image(image):
    segmentation = slic(image, FLAGS.slic_num_segments, FLAGS.slic_compactness,
                        FLAGS.slic_max_iterations, FLAGS.slic_sigma)
    points, adj, mass = segmentation_adjacency(segmentation,
                                               FLAGS.graph_connectivity)
    features = feature_extraction_minimal(segmentation, image)

    adjs_dist, adjs_rad, perm = coarsen_embedded_adj(
        points, mass, adj, levels=4, locale=FLAGS.locale_normalization)

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


def preprocess_images(images):
    all_adjs_1 = []
    all_adjs_2 = []
    all_features = []
    for i in xrange(images.shape[0]):
        adjs_1, adjs_2, features = preprocess_image(images[i])
        all_adjs_1.append(adjs_1)
        all_adjs_2.append(adjs_2)
        all_features.append(features)

    return all_adjs_1, all_adjs_2, all_features


placeholders = {
    'features': [
        tf.placeholder(tf.float32, [None, NUM_FEATURES_MINIMAL],
                       'features_{}'.format(i + 1))
        for i in xrange(FLAGS.batch_size)
    ],
    'adjacency_1': [[
        tf.sparse_placeholder(
            tf.float32, name='adjacency_1_{}_{}'.format(i + 1, j + 1))
        for j in xrange(FLAGS.num_partitions)
    ] for i in xrange(FLAGS.batch_size)],
    'adjacency_2': [[
        tf.sparse_placeholder(
            tf.float32, name='adjacency_2_{}_{}'.format(i + 1, j + 1))
        for j in xrange(FLAGS.num_partitions)
    ] for i in xrange(FLAGS.batch_size)],
    'labels':
    tf.placeholder(tf.int32, [FLAGS.batch_size], 'labels'),
    'dropout':
    tf.placeholder(tf.float32, [], 'dropout'),
}


class MNIST(Model):
    def __init__(self, **kwargs):
        super(MNIST, self).__init__(**kwargs)
        self.build()

    def _build(self):
        conv_1_1 = Conv(
            4,
            32,
            self.placeholders['adjacency_1'],
            num_partitions=FLAGS.num_partitions,
            bias=True,
            logging=self.logging)
        pool_1 = MaxPool(size=4, logging=self.logging)
        conv_2_1 = Conv(
            32,
            64,
            self.placeholders['adjacency_2'],
            num_partitions=FLAGS.num_partitions,
            bias=True,
            logging=self.logging)
        pool_2 = MaxPool(size=4, logging=self.logging)
        fixed_pool = FixedPool(logging=self.logging)
        fc_1 = FC(64, 124, logging=self.logging)
        fc_2 = FC(124,
                  10,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [
            conv_1_1, pool_1, conv_2_1, pool_2, fixed_pool, fc_1, fc_2
        ]


model = MNIST(
    placeholders=placeholders,
    learning_rate=FLAGS.learning_rate,
    log_dir=FLAGS.log_dir)
global_step = model.initialize()


def evaluate(images, labels):
    adjs_1, adjs_2, features = preprocess_images(images)

    feed_dict = {
        placeholders['labels']: labels,
        placeholders['dropout']: 0.0,
    }

    feed_dict.update({
        placeholders['features'][i]: features[i]
        for i in xrange(FLAGS.batch_size)
    })

    feed_dict.update({
        placeholders['adjacency_1'][i][j]: adjs_1[i][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_2'][i][j]: adjs_2[i][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })

    return model.evaluate(feed_dict)


for step in xrange(global_step, FLAGS.max_steps):
    train_images, train_labels = data.next_train_batch(FLAGS.batch_size)
    train_adjs_1, train_adjs_2, train_features = preprocess_images(
        train_images)

    train_feed_dict = {
        placeholders['labels']: train_labels,
        placeholders['dropout']: FLAGS.dropout,
    }

    train_feed_dict.update({
        placeholders['features'][i]: train_features[i]
        for i in xrange(FLAGS.batch_size)
    })

    train_feed_dict.update({
        placeholders['adjacency_1'][i][j]: train_adjs_1[i][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    train_feed_dict.update({
        placeholders['adjacency_2'][i][j]: train_adjs_2[i][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })

    duration = model.train(train_feed_dict, step)

    if step % FLAGS.display_step == 0:
        # Evaluate on training and validation set.
        train_loss, train_acc, _ = evaluate(train_images, train_labels)

        val_images, val_labels = data.next_validation_batch(FLAGS.batch_size)
        val_loss, val_acc, _ = evaluate(val_images, val_labels)

        # Print results.
        print(', '.join([
            'Step: {}'.format(step),
            'train_loss={:.5f}'.format(train_loss),
            'train_acc={:.5f}'.format(train_acc),
            'time={:.2f}s'.format(duration),
            'val_loss={:.5f}'.format(val_loss),
            'val_acc={:.5f}'.format(val_acc),
        ]))

print('Optimization finished!')

# Evaluate on test set.
num_iterations = data.num_test_examples // FLAGS.batch_size
test_loss, test_acc, test_duration = (0, 0, 0)
for i in xrange(num_iterations):
    test_images, test_labels = data.next_test_batch(FLAGS.batch_size)
    test_single_loss, test_single_acc, test_single_duration = evaluate(
        test_images, test_labels)
    test_loss += test_single_loss
    test_acc += test_single_acc
    test_duration += test_single_duration

print('Test set results: cost={:.5f}, accuracy={:.5f}, time={:.2f}s'.format(
    test_loss / num_iterations, test_acc / num_iterations, test_duration))
