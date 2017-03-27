from __future__ import print_function
from __future__ import division

from six.moves import xrange

import tensorflow as tf

from lib.datasets.mnist import MNIST
from lib.model.model import Model
from lib.layer.conv2d import Conv2d as Conv
from lib.layer.max_pool2d import MaxPool2d as MaxPool
from lib.layer.fixed_mean_pool import FixedMeanPool as FixedPool
from lib.layer.fc import FC

flags = tf.app.flags
FLAGS = flags.FLAGS
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

placeholders = {
    'features':
    tf.placeholder(tf.float32,
                   [FLAGS.batch_size, data.height, data.width,
                    data.channels], 'features'),
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
        conv_1 = Conv(1, 32, size=5, stride=1, logging=self.logging)
        pool_1 = MaxPool(size=2, stride=2, logging=self.logging)
        conv_2 = Conv(32, 64, size=5, stride=1, logging=self.logging)
        pool_2 = MaxPool(size=2, stride=2, logging=self.logging)
        fixed_pool = FixedPool(logging=self.logging)
        fc_1 = FC(64,
                  1024,
                  logging=self.logging)
        fc_2 = FC(1024,
                  10,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [conv_1, pool_1, conv_2, pool_2, fixed_pool, fc_1, fc_2]


model = MNISTModel(
    placeholders=placeholders,
    learning_rate=FLAGS.learning_rate,
    log_dir=FLAGS.log_dir)
global_step = model.initialize()


def evaluate(features, labels):
    feed_dict = {
        placeholders['features']: features,
        placeholders['labels']: labels,
        placeholders['dropout']: 0.0,
    }
    return model.evaluate(feed_dict)


for step in xrange(global_step, FLAGS.max_steps):
    train_features, train_labels = data.next_train_batch(FLAGS.batch_size)
    train_feed_dict = {
        placeholders['features']: train_features,
        placeholders['labels']: train_labels,
        placeholders['dropout']: FLAGS.dropout,
    }

    duration = model.train(train_feed_dict, step)

    if step % FLAGS.display_step == 0:
        # Evaluate on training and validation set.
        train_loss, train_acc, _ = evaluate(train_features, train_labels)

        val_features, val_labels = data.next_validation_batch(
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
num_iterations = data.num_test_examples // FLAGS.batch_size
test_loss, test_acc, test_duration = (0, 0, 0)
for i in xrange(num_iterations):
    test_features, test_labels = data.next_test_batch(FLAGS.batch_size)
    test_single_loss, test_single_acc, test_single_duration = evaluate(
        test_features, test_labels)
    test_loss += test_single_loss
    test_acc += test_single_acc
    test_duration += test_single_duration

print('Test set results: cost={:.5f}, accuracy={:.5f}, time={:.2f}s'.format(
    test_loss / num_iterations, test_acc / num_iterations, test_duration))
