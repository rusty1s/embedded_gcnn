from __future__ import print_function
from __future__ import division

from six.moves import xrange
import time

import tensorflow as tf

from lib.datasets import PascalVOC as Data
from lib.model import Model as BaseModel
from lib.layer import Conv2d as Conv, MaxPool, FC

DATA_DIR = 'data/pascal_voc'

LEARNING_RATE = 0.0001
TRAIN_DIR = None
LOG_DIR = 'data/summaries/pascal_conv2d'

DROPOUT = 0.5
BATCH_SIZE = 32
MAX_STEPS = 50000
DISPLAY_STEP = 10
SAVE_STEP = 250

data = Data(DATA_DIR)

placeholders = {
    'features':
    tf.placeholder(tf.float32,
                   [None, data.width, data.height, data.num_channels],
                   'features'),
    'labels':
    tf.placeholder(tf.uint8, [None, data.num_classes], 'labels'),
    'dropout':
    tf.placeholder(tf.float32, [], 'dropout'),
}


class Model(BaseModel):
    def _build(self):
        conv_1_1 = Conv(data.num_channels, 32, logging=self.logging)
        # conv_1_2 = Conv(32, 32, logging=self.logging)
        max_pool_1 = MaxPool(size=2)
        conv_2_1 = Conv(32, 64, logging=self.logging)
        # conv_2_2 = Conv(64, 64, logging=self.logging)
        max_pool_2 = MaxPool(size=2)
        conv_3_1 = Conv(64, 128, logging=self.logging)
        # conv_3_2 = Conv(128, 128, logging=self.logging)
        max_pool_3 = MaxPool(size=2)
        conv_4_1 = Conv(128, 256, logging=self.logging)
        # conv_4_2 = Conv(256, 256, logging=self.logging)
        max_pool_4 = MaxPool(size=2)
        # conv_5_1 = Conv(256, 512, logging=self.logging)
        # conv_5_2 = Conv(512, 512, logging=self.logging)
        # max_pool_5 = MaxPool(size=2)
        fc_1 = FC(14 * 14 * 256, 1024, logging=self.logging)
        fc_2 = FC(
            1024,
            256,
            # dropout=self.placeholders['dropout'],
            logging=self.logging)
        fc_3 = FC(256,
                  data.num_classes,
                  act=lambda x: x,
                  bias=False,
                  logging=self.logging)

        # self.layers = [
        #     conv_1_1, conv_1_2, max_pool_1, conv_2_1, conv_2_2, max_pool_2,
        #     conv_3_1, conv_3_2, max_pool_3, conv_4_1, conv_4_2, max_pool_4,
        #     conv_5_1, conv_5_2, max_pool_5, fc_1, fc_2, fc_3
        # ]
        self.layers = [
            conv_1_1, max_pool_1, conv_2_1, max_pool_2, conv_3_1, max_pool_3,
            conv_4_1, max_pool_4, fc_1, fc_2, fc_3
        ]


model = Model(
    placeholders=placeholders,
    isMultilabel=True,
    learning_rate=LEARNING_RATE,
    train_dir=TRAIN_DIR,
    log_dir=LOG_DIR)

model.build()
global_step = model.initialize()


def feed_dict_with_batch(images, labels, dropout=0):
    return {
        placeholders['features']: images,
        placeholders['labels']: labels,
        placeholders['dropout']: DROPOUT,
    }


try:
    for step in xrange(global_step, MAX_STEPS):
        t_pre = time.process_time()
        images, labels = data.train.next_batch(BATCH_SIZE, shuffle=True)
        feed_dict = feed_dict_with_batch(images, labels, DROPOUT)
        t_pre = time.process_time() - t_pre

        t_train = model.train(feed_dict, step)

        if step % DISPLAY_STEP == 0:
            # Evaluate on training and validation set with zero dropout.
            feed_dict.update({model.placeholders['dropout']: 0})
            images, labels = data.val.next_batch(BATCH_SIZE, shuffle=True)
            val_feed_dict = feed_dict_with_batch(images, labels)

            train_info = model.evaluate(feed_dict, step, 'train')
            val_info = model.evaluate(val_feed_dict, step, 'val')

            log = 'step={}, '.format(step)
            log += 'time={:.2f}s + {:.2f}s, '.format(t_pre, t_train)
            log += 'train_loss={:.5f}, '.format(train_info[0])
            log += 'train_acc={:.5f}, '.format(train_info[1])
            log += 'train_precision={:.5f}, '.format(train_info[2])
            log += 'train_recall={:.5f}, '.format(train_info[3])
            log += 'val_loss={:.5f}, '.format(val_info[0])
            log += 'val_acc={:.5f}, '.format(val_info[1])
            log += 'val_precision={:.5f}, '.format(val_info[2])
            log += 'val_recall={:.5f}'.format(val_info[3])

            print(log)

        if step % SAVE_STEP == 0:
            model.save()

except KeyboardInterrupt:
    print()

print('Optimization finished!')
print('Evaluate on test set. This can take a few minutes.')

try:
    num_steps = data.test.num_examples // BATCH_SIZE
    test_info = [0, 0, 0, 0]

    for i in xrange(num_steps):
        images, labels = data.test.next_batch(BATCH_SIZE, shuffle=False)
        feed_dict = feed_dict_with_batch(images, labels)

        batch_info = model.evaluate(feed_dict)
        test_info = [a + b for a, b in zip(test_info, batch_info)]

    log = 'Test results: '
    log += 'loss={:.5f}, '.format(test_info[0] / num_steps)
    log += 'acc={:.5f}, '.format(test_info[1] / num_steps)
    log += 'precision={:.5f}, '.format(test_info[2] / num_steps)
    log += 'recall={:.5f}'.format(test_info[3] / num_steps)

    print(log)

except KeyboardInterrupt:
    print()
    print('Test evaluation aborted.')
