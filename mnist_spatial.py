from __future__ import print_function
from __future__ import division

from six.moves import xrange
import os
import time

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from lib.datasets import MNIST as Data
from lib.model import Model as BaseModel
from lib.segmentation import segmentation_adjacency, extract_features_fixed
# from lib.segmentation import slic_fixed
from lib.segmentation import quickshift_fixed
from lib.layer import SpatialCNN as Conv, FC
from lib.graph import receptive_fields, fill_features
from lib.pipeline import PreprocessedDataset, FileQueue

# SLIC_FEATURES = [4, 5, 6, 7, 8, 18, 20, 21, 22]
QUICKSHIFT_FEATURES = [4, 6, 7, 8, 24, 28, 29, 31, 37]

DATA_DIR = 'data/mnist'

# PREPROCESS_FIRST = 'data/mnist/slic_spatial'
PREPROCESS_FIRST = 'data/mnist/quickshift_spatial'

NODE_SIZE = 25
NODE_STRIDE = 4
DELTA = 3
NEIGHBORHOOD_SIZE = 25
CONNECTIVITY = 8

LEARNING_RATE = 0.001
TRAIN_DIR = None
# LOG_DIR = 'data/summaries/mnist_slic_spatial'
LOG_DIR = 'data/summaries/mnist_quickshift_spatial'
SAVE_STEP = 250

AUGMENT_TRAIN_EXAMPLES = False
DROPOUT = 0.5
BATCH_SIZE = 64
MAX_STEPS = 15000
DISPLAY_STEP = 10
# FORM_FEATURES = SLIC_FEATURES
FORM_FEATURES = QUICKSHIFT_FEATURES
NUM_FEATURES = len(FORM_FEATURES) + 1

data = Data(DATA_DIR)

# segmentation_algorithm = slic_fixed(
#     num_segments=100, compactness=5, max_iterations=10, sigma=0)
segmentation_algorithm = quickshift_fixed(
    ratio=1, kernel_size=2, max_dist=2, sigma=0)

feature_extraction_algorithm = extract_features_fixed(FORM_FEATURES)


def preprocess_spatial_fixed(
        segmentation_algorithm, feature_extraction_algorithm, node_size,
        node_stride, delta, neighborhood_size, connectivity):
    def _preprocess(image):
        segmentation = segmentation_algorithm(image)
        adj, points, _ = segmentation_adjacency(segmentation, connectivity)
        features = feature_extraction_algorithm(segmentation, image)
        StandardScaler(copy=False).fit_transform(features)

        fields = receptive_fields(points, adj, node_size, node_stride,
                                  neighborhood_size, delta)
        return fill_features(fields, features)

    return _preprocess


preprocess_algorithm = preprocess_spatial_fixed(
    segmentation_algorithm, feature_extraction_algorithm, NODE_SIZE,
    NODE_STRIDE, DELTA, NEIGHBORHOOD_SIZE, CONNECTIVITY)

# Generate preprocessed dataset.
data.train = PreprocessedDataset(
    os.path.join(PREPROCESS_FIRST, 'train'), data.train, preprocess_algorithm)
data.val = PreprocessedDataset(
    os.path.join(PREPROCESS_FIRST, 'val'), data.val, preprocess_algorithm)
data.test = PreprocessedDataset(
    os.path.join(PREPROCESS_FIRST, 'test'), data.test, preprocess_algorithm)

capacity = 10 * BATCH_SIZE
train_queue = FileQueue(data.train, BATCH_SIZE, capacity, shuffle=True)
val_queue = FileQueue(data.val, BATCH_SIZE, capacity, shuffle=True)
test_queue = FileQueue(data.test, BATCH_SIZE, capacity, shuffle=False)

placeholders = {
    'features':
    tf.placeholder(tf.float32,
                   [None, NODE_SIZE, NEIGHBORHOOD_SIZE,
                    NUM_FEATURES], 'features'),
    'labels':
    tf.placeholder(tf.uint8, [None, data.num_classes], 'labels'),
    'dropout':
    tf.placeholder(tf.float32, [], 'dropout'),
}


class Model(BaseModel):
    def _build(self):
        conv_1 = Conv(
            NUM_FEATURES, 64, NEIGHBORHOOD_SIZE, logging=self.logging)
        fc_1 = FC(NODE_SIZE * 64, 256, logging=self.logging)
        fc_2 = FC(
            256,
            data.num_classes,
            act=lambda x: x,
            bias=False,
            dropout=self.placeholders['dropout'],
            logging=self.logging)

        self.layers = [conv_1, fc_1, fc_2]


model = Model(
    placeholders=placeholders,
    learning_rate=LEARNING_RATE,
    train_dir=TRAIN_DIR,
    log_dir=LOG_DIR)

model.build()
global_step = model.initialize()


def feed_dict_with_batch(features, labels, dropout=0):
    return {
        placeholders['features']: features,
        placeholders['labels']: labels,
        placeholders['dropout']: DROPOUT,
    }


try:
    for step in xrange(global_step, MAX_STEPS):
        t_pre = time.process_time()
        batch = train_queue.dequeue()
        feed_dict = feed_dict_with_batch(batch[0], batch[1], DROPOUT)
        t_pre = time.process_time() - t_pre

        t_train = model.train(feed_dict, step)

        if step % DISPLAY_STEP == 0:
            # Evaluate on training and validation set with zero dropout.
            feed_dict.update({model.placeholders['dropout']: 0})
            train_info = model.evaluate(feed_dict, step, 'train')
            batch = val_queue.dequeue()
            feed_dict = feed_dict_with_batch(batch[0], batch[1], DROPOUT)
            val_info = model.evaluate(feed_dict, step, 'val')

            log = 'step={}, '.format(step)
            log += 'time={:.2f}s + {:.2f}s, '.format(t_pre, t_train)
            log += 'train_loss={:.5f}, '.format(train_info[0])
            log += 'train_acc={:.5f}, '.format(train_info[1])
            log += 'val_loss={:.5f}, '.format(val_info[0])
            log += 'val_acc={:.5f}'.format(val_info[1])
            print(log)

        if step % SAVE_STEP == 0:
            model.save()

except KeyboardInterrupt:
    print()

print('Optimization finished!')
print('Evaluate on test set. This can take a few minutes.')

try:
    num_steps = data.test.num_examples // BATCH_SIZE
    test_info = [0, 0]

    for i in xrange(num_steps):
        batch = test_queue.dequeue()
        feed_dict = feed_dict_with_batch(batch[0], batch[1], DROPOUT)

        batch_info = model.evaluate(feed_dict)
        test_info = [a + b for a, b in zip(test_info, batch_info)]

    log = 'Test results: '
    log += 'loss={:.5f}, '.format(test_info[0] / num_steps)
    log += 'acc={:.5f}, '.format(test_info[1] / num_steps)

    print(log)

except KeyboardInterrupt:
    print()
    print('Test evaluation aborted.')
