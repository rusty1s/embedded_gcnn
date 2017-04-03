from __future__ import print_function
from __future__ import division

from six.moves import xrange

import numpy as np
import tensorflow as tf

from lib.datasets.dataset import Datasets
from lib.datasets.cifar_10 import Cifar10
from lib.segmentation.algorithm import felzenszwalb
from lib.segmentation.adjacency import segmentation_adjacency
from lib.segmentation.feature_extraction import (feature_extraction,
                                                 NUM_FEATURES)
from lib.graph.embedding import partition_embedded_adj
from lib.graph.embedded_coarsening import coarsen_embedded_adj
from lib.graph.preprocess import preprocess_adj
from lib.graph.sparse import sparse_to_tensor
from lib.graph.distortion import perm_features
from lib.model.model import Model
from lib.layer.partitioned_gcnn import PartitionedGCNN as Conv
from lib.layer.max_pool_gcnn import MaxPoolGCNN as MaxPool
from lib.layer.fixed_mean_pool import FixedMeanPool as AveragePool
from lib.layer.fc import FC

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('locale_normalization', False,
                     'Whether to normalize each adjacency locally.')
flags.DEFINE_float('stddev', 1, 'Standard deviation for gaussian invert.')
flags.DEFINE_integer('graph_connectivity', 1,
                     '''The connectivity between pixels in the segmentation. A
                     connectivity of 1 corresponds to immediate neighbors up,
                     down, left and right, while a connectivity of 2 also
                     includes diagonal neighbors.''')
flags.DEFINE_float('scale', 10, 'Higher means larger clusters.')
flags.DEFINE_integer('min_size', 2,
                     'Minimum component size. Enforced using postprocessing.')
flags.DEFINE_float('sigma', 2,
                   'Width of gaussian smoothing kernel for preprocessing.')
flags.DEFINE_integer('num_partitions', 8,
                     '''The number of partitions of each graph corresponding to
                     the number of weights for each convolution.''')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 64, 'How many inputs to process at once.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('data_dir', 'data/cifar_10/felzenszwalb/input',
                    'Directory for storing input data.')
flags.DEFINE_string('log_dir', 'data/cifar_10/summaries/felzenszwalb',
                    'Summaries log directory.')
flags.DEFINE_integer('display_step', 5,
                     'How many steps to print logging after.')


class Cifar10Felzenszwalb(Datasets):
    def __init__(self):
        cifar_10 = Cifar10('data/cifar_10')
        super(Cifar10Felzenszwalb, self).__init__(
            cifar_10.train,
            cifar_10.validation,
            cifar_10.test,
            preprocess=True,
            data_dir=None)

    def _preprocess(self, image):
        segmentation = felzenszwalb(image, FLAGS.scale, FLAGS.min_size,
                                    FLAGS.sigma)
        points, adj, mass = segmentation_adjacency(segmentation,
                                                   FLAGS.graph_connectivity)

        adjs_dist, adjs_rad, perm = coarsen_embedded_adj(
            points,
            mass,
            adj,
            levels=4,
            locale=FLAGS.locale_normalization,
            stddev=FLAGS.stddev)

        features = feature_extraction(segmentation, image)
        features = perm_features(features, perm)

        adjs_1 = partition_embedded_adj(
            adjs_dist[0],
            adjs_rad[0],
            num_partitions=FLAGS.num_partitions,
            offset=0.125 * np.pi)
        adjs_2 = partition_embedded_adj(
            adjs_dist[1],
            adjs_rad[1],
            num_partitions=FLAGS.num_partitions,
            offset=0.125 * np.pi)
        adjs_3 = partition_embedded_adj(
            adjs_dist[2],
            adjs_rad[2],
            num_partitions=FLAGS.num_partitions,
            offset=0.125 * np.pi)
        adjs_4 = partition_embedded_adj(
            adjs_dist[3],
            adjs_rad[3],
            num_partitions=FLAGS.num_partitions,
            offset=0.125 * np.pi)

        adjs_1 = [sparse_to_tensor(preprocess_adj(adj)) for adj in adjs_1]
        adjs_2 = [sparse_to_tensor(preprocess_adj(adj)) for adj in adjs_2]
        adjs_3 = [sparse_to_tensor(preprocess_adj(adj)) for adj in adjs_3]
        adjs_4 = [sparse_to_tensor(preprocess_adj(adj)) for adj in adjs_4]

        return {
            'adjs_1': adjs_1,
            'adjs_2': adjs_2,
            'adjs_3': adjs_3,
            'adjs_4': adjs_4,
            'features': features
        }


placeholders = {
    'features': [
        tf.placeholder(tf.float32, [None, NUM_FEATURES],
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
    'adjacency_3': [[
        tf.sparse_placeholder(
            tf.float32, name='adjacency_3_{}_{}'.format(i + 1, j + 1))
        for j in xrange(FLAGS.num_partitions)
    ] for i in xrange(FLAGS.batch_size)],
    'adjacency_4': [[
        tf.sparse_placeholder(
            tf.float32, name='adjacency_4_{}_{}'.format(i + 1, j + 1))
        for j in xrange(FLAGS.num_partitions)
    ] for i in xrange(FLAGS.batch_size)],
    'labels':
    tf.placeholder(tf.int32, [FLAGS.batch_size, 10], 'labels'),
    'dropout':
    tf.placeholder(tf.float32, [], 'dropout'),
}


class Cifar10Model(Model):
    def __init__(self, **kwargs):
        super(Cifar10Model, self).__init__(**kwargs)
        self.build()

    def _build(self):
        conv_1_1 = Conv(
            NUM_FEATURES,
            64,
            self.placeholders['adjacency_1'],
            num_partitions=FLAGS.num_partitions,
            weight_stddev=0.05,
            bias=True,
            bias_constant=0,
            logging=self.logging)
        conv_1_2 = Conv(
            64,
            64,
            self.placeholders['adjacency_1'],
            num_partitions=FLAGS.num_partitions,
            weight_stddev=0.05,
            bias=True,
            bias_constant=0,
            logging=self.logging)
        pool_1 = MaxPool(size=2, logging=self.logging)
        conv_2_1 = Conv(
            64,
            128,
            self.placeholders['adjacency_2'],
            num_partitions=FLAGS.num_partitions,
            weight_stddev=0.05,
            bias=True,
            bias_constant=0,
            logging=self.logging)
        conv_2_2 = Conv(
            128,
            128,
            self.placeholders['adjacency_2'],
            num_partitions=FLAGS.num_partitions,
            weight_stddev=0.05,
            bias=True,
            bias_constant=0,
            logging=self.logging)
        pool_2 = MaxPool(size=2, logging=self.logging)
        conv_3_1 = Conv(
            128,
            256,
            self.placeholders['adjacency_3'],
            num_partitions=FLAGS.num_partitions,
            weight_stddev=0.05,
            bias=True,
            bias_constant=0,
            logging=self.logging)
        conv_3_2 = Conv(
            256,
            256,
            self.placeholders['adjacency_3'],
            num_partitions=FLAGS.num_partitions,
            weight_stddev=0.05,
            bias=True,
            bias_constant=0,
            logging=self.logging)
        pool_3 = MaxPool(size=2, logging=self.logging)
        conv_4_1 = Conv(
            256,
            512,
            self.placeholders['adjacency_4'],
            num_partitions=FLAGS.num_partitions,
            weight_stddev=0.05,
            bias=True,
            bias_constant=0,
            logging=self.logging)
        conv_4_2 = Conv(
            512,
            512,
            self.placeholders['adjacency_4'],
            num_partitions=FLAGS.num_partitions,
            weight_stddev=0.05,
            bias=True,
            bias_constant=0,
            logging=self.logging)
        pool_4 = MaxPool(size=2, logging=self.logging)
        average_pool = AveragePool()
        fc_1 = FC(512,
                  384,
                  weight_stddev=0.04,
                  # weight_decay=0.005,
                  bias=True,
                  bias_constant=0.1,
                  logging=self.logging)
        fc_2 = FC(384,
                  192,
                  weight_stddev=0.04,
                  # weight_decay=0.005,
                  bias=True,
                  bias_constant=0.1,
                  logging=self.logging)
        fc_3 = FC(192,
                  10,
                  weight_stddev=1 / 192,
                  bias=True,
                  bias_constant=0.1,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [
            conv_1_1, conv_1_2, pool_1, conv_2_1, conv_2_2, pool_2, conv_3_1,
            conv_3_2, pool_3, conv_4_1, conv_4_2, pool_4, average_pool, fc_1,
            fc_2, fc_3
        ]


data = Cifar10Felzenszwalb()
model = Cifar10Model(
    placeholders=placeholders,
    learning_rate=FLAGS.learning_rate,
    log_dir=FLAGS.log_dir)
global_step = model.initialize()


def evaluate(data_batch, labels_batch):
    feed_dict = {
        placeholders['labels']: labels_batch,
        placeholders['dropout']: 0.0,
    }

    feed_dict.update({
        placeholders['features'][i]: data_batch[i]['features']
        for i in xrange(FLAGS.batch_size)
    })

    feed_dict.update({
        placeholders['adjacency_1'][i][j]: data_batch[i]['adjs_1'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_2'][i][j]: data_batch[i]['adjs_2'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_3'][i][j]: data_batch[i]['adjs_3'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_4'][i][j]: data_batch[i]['adjs_4'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })

    return model.evaluate(feed_dict)


for step in xrange(global_step, FLAGS.max_steps):
    data_batch, labels_batch = data.train.next_batch(FLAGS.batch_size)

    feed_dict = {
        placeholders['labels']: labels_batch,
        placeholders['dropout']: FLAGS.dropout,
    }

    feed_dict.update({
        placeholders['features'][i]: data_batch[i]['features']
        for i in xrange(FLAGS.batch_size)
    })

    feed_dict.update({
        placeholders['adjacency_1'][i][j]: data_batch[i]['adjs_1'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_2'][i][j]: data_batch[i]['adjs_2'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_3'][i][j]: data_batch[i]['adjs_3'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_4'][i][j]: data_batch[i]['adjs_4'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })

    duration = model.train(feed_dict, step)

    if step % FLAGS.display_step == 0:
        # Evaluate on training and validation set.
        train_loss, train_acc, _ = evaluate(data_batch, labels_batch)

        data_batch, labels_batch = data.validation.next_batch(FLAGS.batch_size)
        val_loss, val_acc, _ = evaluate(data_batch, labels_batch)

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
num_iterations = data.test.num_examples // FLAGS.batch_size
test_loss, test_acc, test_duration = (0, 0, 0)
for i in xrange(num_iterations):
    data_batch, labels_batch = data.test.next_batch(
        FLAGS.batch_size, shuffle=False)
    test_l, test_a, test_d = evaluate(data_batch, labels_batch)
    test_loss += test_l
    test_acc += test_a
    test_duration += test_d

print('Test set results: cost={:.5f}, accuracy={:.5f}, time={:.2f}s'.format(
    test_loss / num_iterations, test_acc / num_iterations, test_duration))
