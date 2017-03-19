from __future__ import print_function
from __future__ import division

from six.moves import xrange

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from lib.graph.adjacency import grid_adj, normalize_adj, invert_adj
from lib.graph.coarsening import coarsen_adj
from lib.graph.preprocess import preprocess_adj
from lib.graph.sparse import sparse_to_tensor
from lib.graph.distortion import perm_batch_of_features
from lib.model.model import Model
from lib.layer.gcnn import GCNN as Conv
from lib.layer.max_pool_gcnn import MaxPoolGCNN as MaxPool
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
flags.DEFINE_string('log_dir', 'data/mnist/summaries/gcnn',
                    'Summaries log directory.')
flags.DEFINE_integer('display_step', 10,
                     'How many steps to print logging after.')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)

# Generate data.
adj = grid_adj([28, 28], FLAGS.grid_connectivity)
adj = normalize_adj(adj)
adj = invert_adj(adj)
adjs, perm = coarsen_adj(adj, levels=4)
adjs = [adjs[0], adjs[2]]
n_1 = adjs[0].shape[0]
n_2 = adjs[1].shape[0]

l = []
for adj in adjs:
    adj = preprocess_adj(adj)
    adj = sparse_to_tensor(adj)
    l.append(adj)
adjs = l

placeholders = {
    'features':
    tf.placeholder(tf.float32, [FLAGS.batch_size, n_1, 1], 'features'),
    'adjacency_1':
    tf.sparse_placeholder(tf.float32, name='adjacency_1'),
    'adjacency_2':
    tf.sparse_placeholder(tf.float32, name='adjacency_2'),
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
        conv_1 = Conv(
            1,
            32,
            self.placeholders['adjacency_1'],
            bias=False,
            logging=self.logging)
        max_pool_1 = MaxPool(size=4, logging=self.logging)
        conv_2 = Conv(
            32,
            64,
            self.placeholders['adjacency_2'],
            bias=False,
            logging=self.logging)
        max_pool_2 = MaxPool(size=4, logging=self.logging)
        fc_1 = FC(n_1 // 4 // 4 * 64, 1024, logging=self.logging)
        fc_2 = FC(1024,
                  10,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [conv_1, max_pool_1, conv_2, max_pool_2, fc_1, fc_2]


model = MNIST(
    placeholders=placeholders,
    learning_rate=FLAGS.learning_rate,
    log_dir=FLAGS.log_dir)
global_step = model.initialize()


def preprocess_features(features):
    features = np.reshape(features, (features.shape[0], features.shape[1], 1))
    return perm_batch_of_features(features, perm)


def evaluate(features, labels):
    features = preprocess_features(features)
    feed_dict = {
        placeholders['features']: features,
        placeholders['adjacency_1']: adjs[0],
        placeholders['adjacency_2']: adjs[1],
        placeholders['labels']: labels,
        placeholders['dropout']: 0.0,
    }

    return model.evaluate(feed_dict)


for step in xrange(global_step, FLAGS.max_steps):
    train_features, train_labels = mnist.train.next_batch(FLAGS.batch_size)
    train_preprocessed_features = preprocess_features(train_features)

    train_feed_dict = {
        placeholders['features']: train_preprocessed_features,
        placeholders['adjacency_1']: adjs[0],
        placeholders['adjacency_2']: adjs[1],
        placeholders['labels']: train_labels,
        placeholders['dropout']: FLAGS.dropout,
    }

    duration = model.train(train_feed_dict, step)

    if step % FLAGS.display_step == 0:
        # Evaluate on training and validation set.
        train_loss, train_acc, _ = evaluate(train_features, train_labels)

        val_features, val_labels = mnist.validation.next_batch(
            FLAGS.batch_size)
        val_loss, val_acc, _ = evaluate(val_features, val_labels)

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
    test_features, test_labels = mnist.test.next_batch(FLAGS.batch_size)
    test_single_loss, test_single_acc, test_single_duration = evaluate(
        test_features, test_labels)
    test_loss += test_single_loss
    test_acc += test_single_acc
    test_duration += test_single_duration

print('Test set results: cost={:.5f}, accuracy={:.5f}, time={:.2f}s'.format(
    test_loss / num_iterations, test_acc / num_iterations, test_duration))
