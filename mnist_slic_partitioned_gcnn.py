from __future__ import print_function
from __future__ import division

from six.moves import xrange

import numpy as np
import tensorflow as tf

from lib.datasets.dataset import Datasets
from lib.datasets.mnist import MNIST
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
from lib.layer.fixed_mean_pool import FixedMeanPool as AveragePool
from lib.layer.fc import FC

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('locale_normalization', False,
                     '''Whether to normalize each adjacency locally.''')
flags.DEFINE_float('stddev', 1, 'Standard deviation for gaussian invert.')
flags.DEFINE_integer('graph_connectivity', 1,
                     '''The connectivity between pixels in the segmentation. A
                     connectivity of 1 corresponds to immediate neighbors up,
                     down, left and right, while a connectivity of 2 also
                     includes diagonal neighbors.''')
flags.DEFINE_integer('slic_num_segments', 100,
                     '''Approximate number of labels in the segmented output
                     image.''')
flags.DEFINE_float('slic_compactness', 10,
                   '''Balances color proximity and space proximity. Higher
                   values give more weight to space proximity, making
                   superpixel shapes more square.''')
flags.DEFINE_integer('slic_max_iterations', 10,
                     'Maximum number of iterations of k-means')
flags.DEFINE_float('slic_sigma', 0,
                   'Width of gaussian smoothing kernel for preprocessing.')
flags.DEFINE_integer('num_partitions', 8,
                     '''The number of partitions of each graph corresponding to
                     the number of weights for each convolution.''')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 64, 'How many inputs to process at once.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('data_dir', 'data/mnist/slic/input',
                    'Directory for storing input data.')
flags.DEFINE_string('log_dir',
                    'data/mnist/summaries/mnist_slic_partitioned_gcnn',
                    'Summaries log directory.')
flags.DEFINE_integer('display_step', 5,
                     'How many steps to print logging after.')


class MNISTSlic(Datasets):
    def __init__(self):
        mnist = MNIST(FLAGS.data_dir)
        super(MNISTSlic, self).__init__(
            mnist.train,
            mnist.validation,
            mnist.test,
            preprocess=True,
            data_dir=FLAGS.data_dir)

    def _preprocess(self, image):
        segmentation = slic(image, FLAGS.slic_num_segments,
                            FLAGS.slic_compactness, FLAGS.slic_max_iterations,
                            FLAGS.slic_sigma)
        points, adj, mass = segmentation_adjacency(segmentation,
                                                   FLAGS.graph_connectivity)

        adjs_dist, adjs_rad, perm = coarsen_embedded_adj(
            points,
            mass,
            adj,
            levels=4,
            locale=FLAGS.locale_normalization,
            stddev=FLAGS.stddev)

        features = feature_extraction_minimal(segmentation, image)
        features = perm_features(features, perm)

        return {
            'adjs_dist': adjs_dist,
            'adjs_rad': adjs_rad,
            'features': features
        }

    def _postprocess(self, data_batch):
        data_batch_new = []
        for i in xrange(len(data_batch)):
            data = data_batch[i]

            adjs_1 = partition_embedded_adj(
                data['adjs_dist'][0],
                data['adjs_rad'][0],
                num_partitions=FLAGS.num_partitions,
                offset=0.125 * np.pi)
            adjs_2 = partition_embedded_adj(
                data['adjs_dist'][1],
                data['adjs_rad'][1],
                num_partitions=FLAGS.num_partitions,
                offset=0.125 * np.pi)
            adjs_3 = partition_embedded_adj(
                data['adjs_dist'][2],
                data['adjs_rad'][2],
                num_partitions=FLAGS.num_partitions,
                offset=0.125 * np.pi)
            adjs_4 = partition_embedded_adj(
                data['adjs_dist'][3],
                data['adjs_rad'][3],
                num_partitions=FLAGS.num_partitions,
                offset=0.125 * np.pi)

            adjs_1 = [sparse_to_tensor(preprocess_adj(adj)) for adj in adjs_1]
            adjs_2 = [sparse_to_tensor(preprocess_adj(adj)) for adj in adjs_2]
            adjs_3 = [sparse_to_tensor(preprocess_adj(adj)) for adj in adjs_3]
            adjs_4 = [sparse_to_tensor(preprocess_adj(adj)) for adj in adjs_4]

            data_batch_new.append({
                '1': adjs_1,
                '2': adjs_2,
                '3': adjs_3,
                '4': adjs_4,
                'features': data['features']
            })

        return data_batch_new


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
    tf.placeholder(tf.int32, [FLAGS.batch_size], 'labels'),
    'dropout':
    tf.placeholder(tf.float32, [], 'dropout'),
}


class MNISTModel(Model):
    def __init__(self, **kwargs):
        super(MNISTModel, self).__init__(**kwargs)
        self.build()

    def _build(self):
        conv_1 = Conv(
            NUM_FEATURES_MINIMAL,
            32,
            self.placeholders['adjacency_1'],
            num_partitions=FLAGS.num_partitions,
            bias=True,
            logging=self.logging)
        pool_1 = MaxPool(size=2, logging=self.logging)
        conv_2 = Conv(
            32,
            64,
            self.placeholders['adjacency_2'],
            num_partitions=FLAGS.num_partitions,
            bias=True,
            logging=self.logging)
        pool_2 = MaxPool(size=2, logging=self.logging)
        conv_3 = Conv(
            64,
            128,
            self.placeholders['adjacency_3'],
            num_partitions=FLAGS.num_partitions,
            bias=True,
            logging=self.logging)
        pool_3 = MaxPool(size=2, logging=self.logging)
        conv_4 = Conv(
            128,
            256,
            self.placeholders['adjacency_4'],
            num_partitions=FLAGS.num_partitions,
            bias=True,
            logging=self.logging)
        pool_4 = MaxPool(size=2, logging=self.logging)
        average_pool = AveragePool()
        fc = FC(256,
                10,
                dropout=self.placeholders['dropout'],
                act=lambda x: x,
                logging=self.logging)

        self.layers = [
            conv_1, pool_1, conv_2, pool_2, conv_3, pool_3, conv_4, pool_4,
            average_pool, fc
        ]


data = MNISTSlic()
model = MNISTModel(
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
        placeholders['adjacency_1'][i][j]: data_batch[i]['1'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_2'][i][j]: data_batch[i]['2'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_3'][i][j]: data_batch[i]['3'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_4'][i][j]: data_batch[i]['4'][j]
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
        placeholders['adjacency_1'][i][j]: data_batch[i]['1'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_2'][i][j]: data_batch[i]['2'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_3'][i][j]: data_batch[i]['3'][j]
        for i in xrange(FLAGS.batch_size) for j in xrange(FLAGS.num_partitions)
    })
    feed_dict.update({
        placeholders['adjacency_4'][i][j]: data_batch[i]['4'][j]
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
    data_batch, labels_batch = data.test.next_batch(FLAGS.batch_size)
    test_l, test_a, test_d = evaluate(data_batch, labels_batch)
    test_loss += test_l
    test_acc += test_a
    test_duration += test_d

print('Test set results: cost={:.5f}, accuracy={:.5f}, time={:.2f}s'.format(
    test_loss / num_iterations, test_acc / num_iterations, test_duration))
