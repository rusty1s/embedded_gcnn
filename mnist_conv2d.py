from __future__ import print_function

from six.moves import xrange

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from lib.model.model import Model
from lib.layer.conv2d import Conv2d as Conv
from lib.layer.max_pool2d import MaxPool2d as MaxPool
from lib.layer.fc import FC

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 128, 'How many inputs to process at once.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('data_dir', 'data/mnist/input',
                    'Directory for storing input data.')
flags.DEFINE_string('log_dir', 'data/mnist/summaries/chebyshev_gcnn',
                    'Summaries log directory.')
flags.DEFINE_integer('display_step', 10,
                     'How many steps to print logging after.')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)

placeholders = {
    'features':
    tf.placeholder(tf.float32, [FLAGS.batch_size, 28, 28, 1], 'features'),
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
        conv_1 = Conv(1, 32, size=5, stride=1, logging=self.logging)
        pool_1 = MaxPool(size=2, stride=2, logging=self.logging)
        conv_2 = Conv(32, 64, size=5, stride=1, logging=self.logging)
        pool_2 = MaxPool(size=2, stride=2, logging=self.logging)
        fc_1 = FC(7 * 7 * 64, 1024, logging=self.logging)
        fc_2 = FC(1024,
                  10,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [conv_1, pool_1, conv_2, pool_2, fc_1, fc_2]


model = MNIST(
    placeholders=placeholders, learning_rate=FLAGS.learning_rate, logging=True)


def preprocess_features(features):
    return np.reshape(features, (features.shape[0], 28, 28, 1))


def evaluate(features, labels):
    features = preprocess_features(features)
    feed_dict = {
        placeholders['features']: features,
        placeholders['labels']: labels,
        placeholders['dropout']: 0.0,
    }

    loss, acc = sess.run([model.loss, model.accuracy], feed_dict)
    return loss, acc


sess = tf.Session()
global_step = model.initialize(sess)
writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

for step in xrange(global_step, FLAGS.max_steps):
    train_features, train_labels = mnist.train.next_batch(FLAGS.batch_size)
    train_preprocessed_features = preprocess_features(train_features)

    train_feed_dict = {
        placeholders['features']: train_preprocessed_features,
        placeholders['labels']: train_labels,
        placeholders['dropout']: FLAGS.dropout,
    }

    _, summary = sess.run([model.train, model.summary], train_feed_dict)
    writer.add_summary(summary, step)

    if step % FLAGS.display_step == 0:
        # Evaluate on training and validation set.
        train_loss, train_acc = evaluate(train_features, train_labels)

        val_features, val_labels = mnist.validation.next_batch(
            FLAGS.batch_size)
        val_loss, val_acc = evaluate(val_features, val_labels)

        # Print results.
        print(', '.join([
            'Step: {}'.format(step),
            'train_loss={:.5f}'.format(train_loss),
            'train_acc={:.5f}'.format(train_acc),
            'val_loss={:.5f}'.format(val_loss),
            'val_acc={:.5f}'.format(val_acc),
        ]))

print('Optimization finished!')

# Evaluate on test set.
test_features = mnist.test.images[:FLAGS.batch_size]
test_labels = mnist.test.labels[:FLAGS.batch_size]
test_loss, test_acc = evaluate(test_features, test_labels)
print('Test set results: cost={:.5f}, accuracy={:.5f}'.format(test_loss,
                                                              test_acc))
