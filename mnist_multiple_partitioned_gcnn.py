from __future__ import print_function
from __future__ import division

from six.moves import xrange

import numpy as np
import tensorflow as tf

from lib.datasets.mnist import MNIST

from lib.graph.embedding import grid_points, partition_embedded_adj
from lib.graph.adjacency import grid_adj
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
flags.DEFINE_integer('grid_connectivity', 8,
                     'Connectivity of the generated grid.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 64, 'How many inputs to process at once.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('data_dir', 'data/mnist/input',
                    'Directory for storing input data.')
flags.DEFINE_string('log_dir',
                    'data/mnist/summaries/multiple_partitioned_gcnn',
                    'Summaries log directory.')
flags.DEFINE_integer('display_step', 10,
                     'How many steps to print logging after.')

data = MNIST(data_dir=FLAGS.data_dir)

# Generate data.
points = grid_points((28, 28))
adj = grid_adj((28, 28), connectivity=FLAGS.grid_connectivity)
mass = np.ones((points.shape[0]))
adjs_dist, adjs_rad, perm = coarsen_embedded_adj(points, mass, adj, levels=8)
n_1 = adjs_dist[0].shape[0]
adjs_1 = partition_embedded_adj(
    adjs_dist[0],
    adjs_rad[0],
    num_partitions=FLAGS.grid_connectivity,
    offset=0.125 * np.pi)
adjs_2 = partition_embedded_adj(
    adjs_dist[2],
    adjs_rad[2],
    num_partitions=FLAGS.grid_connectivity,
    offset=0.125 * np.pi)
adjs_3 = partition_embedded_adj(
    adjs_dist[4],
    adjs_rad[4],
    num_partitions=FLAGS.grid_connectivity,
    offset=0.125 * np.pi)
adjs_4 = partition_embedded_adj(
    adjs_dist[6],
    adjs_rad[6],
    num_partitions=FLAGS.grid_connectivity,
    offset=0.125 * np.pi)
adjs_1 = [sparse_to_tensor(preprocess_adj(a)) for a in adjs_1]
adjs_2 = [sparse_to_tensor(preprocess_adj(a)) for a in adjs_2]
adjs_3 = [sparse_to_tensor(preprocess_adj(a)) for a in adjs_3]
adjs_4 = [sparse_to_tensor(preprocess_adj(a)) for a in adjs_4]


def preprocess_image(image, perm):
    features = np.reshape(image, (-1, 1))
    return adjs_1, adjs_2, adjs_3, adjs_4, perm_features(features, perm)


def preprocess_images(images, perm):
    all_adjs_1 = []
    all_adjs_2 = []
    all_adjs_3 = []
    all_adjs_4 = []
    all_features = []
    for i in xrange(images.shape[0]):
        adjs_1, adjs_2, adjs_3, adjs_4, features = preprocess_image(images[i],
                                                                    perm)
        all_adjs_1.append(adjs_1)
        all_adjs_2.append(adjs_2)
        all_adjs_3.append(adjs_3)
        all_adjs_4.append(adjs_4)
        all_features.append(features)

    return all_adjs_1, all_adjs_2, all_adjs_3, all_adjs_4, all_features


placeholders = {
    'features': [
        tf.placeholder(tf.float32, [None, 1], 'features_{}'.format(i + 1))
        for i in xrange(FLAGS.batch_size)
    ],
    'adjacency_1': [[
        tf.sparse_placeholder(
            tf.float32, name='adjacency_1_{}_{}'.format(i + 1, j + 1))
        for j in xrange(FLAGS.grid_connectivity)
    ] for i in xrange(FLAGS.batch_size)],
    'adjacency_2': [[
        tf.sparse_placeholder(
            tf.float32, name='adjacency_2_{}_{}'.format(i + 1, j + 1))
        for j in xrange(FLAGS.grid_connectivity)
    ] for i in xrange(FLAGS.batch_size)],
    'adjacency_3': [[
        tf.sparse_placeholder(
            tf.float32, name='adjacency_3_{}_{}'.format(i + 1, j + 1))
        for j in xrange(FLAGS.grid_connectivity)
    ] for i in xrange(FLAGS.batch_size)],
    'adjacency_4': [[
        tf.sparse_placeholder(
            tf.float32, name='adjacency_4_{}_{}'.format(i + 1, j + 1))
        for j in xrange(FLAGS.grid_connectivity)
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
            1,
            32,
            self.placeholders['adjacency_1'],
            num_partitions=FLAGS.grid_connectivity,
            bias=True,
            logging=self.logging)
        conv_1_2 = Conv(
            32,
            32,
            self.placeholders['adjacency_1'],
            num_partitions=FLAGS.grid_connectivity,
            bias=True,
            logging=self.logging)
        pool_1 = MaxPool(size=4, logging=self.logging)
        conv_2_1 = Conv(
            32,
            64,
            self.placeholders['adjacency_2'],
            num_partitions=FLAGS.grid_connectivity,
            bias=True,
            logging=self.logging)
        conv_2_2 = Conv(
            64,
            64,
            self.placeholders['adjacency_2'],
            num_partitions=FLAGS.grid_connectivity,
            bias=True,
            logging=self.logging)
        pool_2 = MaxPool(size=4, logging=self.logging)
        conv_3_1 = Conv(
            64,
            128,
            self.placeholders['adjacency_3'],
            num_partitions=FLAGS.grid_connectivity,
            bias=True,
            logging=self.logging)
        conv_3_2 = Conv(
            128,
            128,
            self.placeholders['adjacency_3'],
            num_partitions=FLAGS.grid_connectivity,
            bias=True,
            logging=self.logging)
        pool_3 = MaxPool(size=4, logging=self.logging)
        conv_4_1 = Conv(
            128,
            256,
            self.placeholders['adjacency_4'],
            num_partitions=FLAGS.grid_connectivity,
            bias=True,
            logging=self.logging)
        conv_4_2 = Conv(
            256,
            256,
            self.placeholders['adjacency_4'],
            num_partitions=FLAGS.grid_connectivity,
            bias=True,
            logging=self.logging)
        pool_4 = MaxPool(size=4, logging=self.logging)
        average_pool = AveragePool()
        fc = FC(256,
                10,
                dropout=self.placeholders['dropout'],
                act=lambda x: x,
                logging=self.logging)

        self.layers = [
            conv_1_1, conv_1_2, pool_1, conv_2_1, conv_2_2, pool_2, conv_3_1,
            conv_3_2, pool_3, conv_4_1, conv_4_2, pool_4, average_pool, fc
        ]


model = MNIST(
    placeholders=placeholders,
    learning_rate=FLAGS.learning_rate,
    log_dir=FLAGS.log_dir)
global_step = model.initialize()


def evaluate(images, labels, perm):
    adjs_1, adjs_2, adjs_3, adjs_4, features = preprocess_images(images, perm)
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
        for i in xrange(FLAGS.batch_size)
        for j in xrange(FLAGS.grid_connectivity)
    })
    feed_dict.update({
        placeholders['adjacency_2'][i][j]: adjs_2[i][j]
        for i in xrange(FLAGS.batch_size)
        for j in xrange(FLAGS.grid_connectivity)
    })
    feed_dict.update({
        placeholders['adjacency_3'][i][j]: adjs_3[i][j]
        for i in xrange(FLAGS.batch_size)
        for j in xrange(FLAGS.grid_connectivity)
    })
    feed_dict.update({
        placeholders['adjacency_4'][i][j]: adjs_4[i][j]
        for i in xrange(FLAGS.batch_size)
        for j in xrange(FLAGS.grid_connectivity)
    })

    return model.evaluate(feed_dict)


for step in xrange(global_step, FLAGS.max_steps):
    images, labels = data.next_train_batch(FLAGS.batch_size)
    adjs_1, adjs_2, adjs_3, adjs_4, features = preprocess_images(images, perm)

    train_feed_dict = {
        placeholders['labels']: labels,
        placeholders['dropout']: FLAGS.dropout,
    }

    train_feed_dict.update({
        placeholders['features'][i]: features[i]
        for i in xrange(FLAGS.batch_size)
    })

    train_feed_dict.update({
        placeholders['adjacency_1'][i][j]: adjs_1[i][j]
        for i in xrange(FLAGS.batch_size)
        for j in xrange(FLAGS.grid_connectivity)
    })
    train_feed_dict.update({
        placeholders['adjacency_2'][i][j]: adjs_2[i][j]
        for i in xrange(FLAGS.batch_size)
        for j in xrange(FLAGS.grid_connectivity)
    })
    train_feed_dict.update({
        placeholders['adjacency_3'][i][j]: adjs_3[i][j]
        for i in xrange(FLAGS.batch_size)
        for j in xrange(FLAGS.grid_connectivity)
    })
    train_feed_dict.update({
        placeholders['adjacency_4'][i][j]: adjs_4[i][j]
        for i in xrange(FLAGS.batch_size)
        for j in xrange(FLAGS.grid_connectivity)
    })

    duration = model.train(train_feed_dict, step)

    if step % FLAGS.display_step == 0:
        # Evaluate on training and validation set.
        train_loss, train_acc, _ = evaluate(images, labels, perm)

        images, labels = data.next_validation_batch(FLAGS.batch_size)
        val_loss, val_acc, _ = evaluate(images, labels, perm)

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
num_iterations = 10000 // FLAGS.batch_size
test_loss, test_acc, test_duration = (0, 0, 0)
for i in xrange(num_iterations):
    images, labels = data.next_test_batch(FLAGS.batch_size)
    test_single_loss, test_single_acc, test_single_duration = evaluate(
        images, labels, perm)
    test_loss += test_single_loss
    test_acc += test_single_acc
    test_duration += test_single_duration

print('Test set results: cost={:.5f}, accuracy={:.5f}, time={:.2f}s'.format(
    test_loss / num_iterations, test_acc / num_iterations, test_duration))
