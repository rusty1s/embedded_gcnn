from __future__ import print_function
from __future__ import division

import time
from six.moves import xrange

from lib.datasets.mnist import MNIST
from lib.datasets.queue import PreprocessQueue
from lib.segmentation.algorithm import slic_fixed
from lib.segmentation.feature_extraction import mnist_slic_feature_extraction
from lib.pipeline.preprocess import preprocess_fixed
from lib.model.embedded_placeholder import (embedded_placeholders,
                                            embedded_feed_dict)

from lib.model.model import Model
from lib.layer.embedded_gcnn import EmbeddedGCNN as Conv
from lib.layer.max_pool import MaxPool
from lib.layer.average_pool import AveragePool
from lib.layer.fc import FC

LEARNING_RATE = 0.1
BATCH_SIZE = 64
MAX_STEPS = 2000
DROPOUT = 0.5
DATA_DIR = 'data/mnist'
DISPLAY_STEP = 10

LEVELS = 4
NUM_FEATURES = 6


class MNISTModel(Model):
    def __init__(self, **kwargs):
        super(MNISTModel, self).__init__(**kwargs)
        self.build()

    def _build(self):
        conv_1 = Conv(
            6,
            32,
            adjs_dist=self.placeholders['adj_dist_1'],
            adjs_rad=self.placeholders['adj_rad_1'])
        max_pool_1 = MaxPool(size=2)
        conv_2 = Conv(
            32,
            64,
            adjs_dist=self.placeholders['adj_dist_2'],
            adjs_rad=self.placeholders['adj_rad_2'])
        max_pool_2 = MaxPool(size=2)
        conv_3 = Conv(
            64,
            128,
            adjs_dist=self.placeholders['adj_dist_3'],
            adjs_rad=self.placeholders['adj_rad_3'])
        max_pool_3 = MaxPool(size=2)
        conv_4 = Conv(
            128,
            256,
            adjs_dist=self.placeholders['adj_dist_4'],
            adjs_rad=self.placeholders['adj_rad_4'])
        max_pool_4 = MaxPool(size=2)
        average_pool = AveragePool()
        fc_1 = FC(256, 128)
        fc_2 = FC(128,
                  data.num_labels,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [
            conv_1, max_pool_1, conv_2, max_pool_2, conv_3, max_pool_3, conv_4,
            max_pool_4, average_pool, fc_1, fc_2
        ]


data = MNIST(DATA_DIR)
segmentation_algorithm = slic_fixed(
    num_segments=100, compactness=2, max_iterations=10, sigma=0)
feature_extraction_algorithm = mnist_slic_feature_extraction
preprocess_algorithm = preprocess_fixed(
    segmentation_algorithm, feature_extraction_algorithm, levels=4)
train_queue = PreprocessQueue(
    data.train,
    preprocess_algorithm,
    batch_size=BATCH_SIZE,
    capacity=500,
    shuffle=True)
validation_queue = PreprocessQueue(
    data.validation,
    preprocess_algorithm,
    batch_size=BATCH_SIZE,
    capacity=500,
    shuffle=True)
placeholders = embedded_placeholders(BATCH_SIZE, LEVELS, NUM_FEATURES,
                                     data.num_labels)
model = MNISTModel(placeholders=placeholders, learning_rate=LEARNING_RATE)
global_step = model.initialize()

for step in xrange(global_step, MAX_STEPS):
    t_preprocess = time.process_time()
    batch = train_queue.dequeue()
    train_feed_dict = embedded_feed_dict(placeholders, batch, dropout=0.5)
    t_preprocess = time.process_time() - t_preprocess

    t_train = model.train(train_feed_dict, step)

    if step % DISPLAY_STEP == 0:
        # Evaluate on training and validation set.
        train_feed_dict.update({placeholders['dropout']: 0})
        train_loss, train_acc, _ = model.evaluate(train_feed_dict)

        batch = validation_queue.dequeue()
        validation_feed_dict = embedded_feed_dict(placeholders, batch)
        val_loss, val_acc, _ = model.evaluate(validation_feed_dict)

        # Print results.
        print(', '.join([
            'Step: {}'.format(step),
            'train_loss={:.5f}'.format(train_loss),
            'train_acc={:.5f}'.format(train_acc),
            'time={:.2f}s + {:.2f}s'.format(t_preprocess, t_train),
            'val_loss={:.5f}'.format(val_loss),
            'val_acc={:.5f}'.format(val_acc),
        ]))

train_queue.close()
validation_queue.close()

print('Optimization finished!')

test_queue = PreprocessQueue(
    data.test,
    preprocess_algorithm,
    batch_size=BATCH_SIZE,
    capacity=500,
    shuffle=False)

# Evaluate on test set.
num_iterations = data.test.num_examples // BATCH_SIZE
test_loss, test_acc, test_duration = (0, 0, 0)
for i in xrange(num_iterations):
    batch = test_queue.dequeue()
    test_feed_dict = embedded_feed_dict(placeholders, batch)
    test_single_loss, test_single_acc, test_single_duration = model.evaluate(
        test_feed_dict)
    test_loss += test_single_loss
    test_acc += test_single_acc
    test_duration += test_single_duration

print('Test set results: cost={:.5f}, accuracy={:.5f}, time={:.2f}s'.format(
    test_loss / num_iterations, test_acc / num_iterations, test_duration))

test_queue.close()
