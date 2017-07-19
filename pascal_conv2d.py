from __future__ import print_function
from __future__ import division

from six.moves import xrange
import time

import tensorflow as tf

from lib.datasets import PascalVOC as Data
from lib.model import Model as BaseModel
from lib.layer import ImageAugment, Conv2d, Fire, MaxPool, AveragePool

DATA_DIR = 'data/pascal_voc'

LEARNING_RATE = 0.0001
TRAIN_DIR = None
LOG_DIR = 'data/summaries/pascal_conv2d'
SAVE_STEP = 250

DROPOUT = 0.5
BATCH_SIZE = 64
MAX_STEPS = 50000
DISPLAY_STEP = 10

data = Data(DATA_DIR, fixed_size=224)

placeholders = {
    'features':
    tf.placeholder(tf.float32,
                   [None, data.width, data.height,
                    data.num_channels], 'features'),
    'labels':
    tf.placeholder(tf.uint8, [None, data.num_classes], 'labels'),
    'dropout':
    tf.placeholder(tf.float32, [], 'dropout'),
}


class Model(BaseModel):
    def _build(self):
        augment = ImageAugment()
        conv_1 = Conv2d(
            data.num_channels, 64, size=3, stride=2, logging=self.logging)
        max_pool_1 = MaxPool(3, 2)

        fire_1_1 = Fire(64, 16, 64, logging=self.logging)
        fire_1_2 = Fire(128, 16, 64, logging=self.logging)

        max_pool_2 = MaxPool(3, 2)

        fire_2_1 = Fire(128, 32, 128, logging=self.logging)
        fire_2_2 = Fire(256, 32, 128, logging=self.logging)

        max_pool_3 = MaxPool(3, 2)

        fire_3_1 = Fire(256, 48, 192, logging=self.logging)
        fire_3_2 = Fire(384, 48, 192, logging=self.logging)
        fire_3_3 = Fire(384, 64, 256, logging=self.logging)
        fire_3_4 = Fire(512, 64, 256, logging=self.logging)

        conv_2 = Conv2d(
            512,
            20,
            size=1,
            stride=1,
            bias=False,
            dropout=DROPOUT,
            logging=self.logging)

        avg = AveragePool()

        self.layers = [
            augment, conv_1, max_pool_1, fire_1_1, fire_1_2, max_pool_2,
            fire_2_1, fire_2_2, max_pool_3, fire_3_1, fire_3_2, fire_3_3,
            fire_3_4, conv_2, avg
        ]


model = Model(
    placeholders=placeholders,
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
            train_info = model.evaluate(feed_dict, step, 'train')
            images, labels = data.val.next_batch(BATCH_SIZE, shuffle=True)
            val_feed_dict = feed_dict_with_batch(images, labels)
            val_info = model.evaluate(val_feed_dict, step, 'val')

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
        images, labels = data.test.next_batch(BATCH_SIZE, shuffle=False)
        feed_dict = feed_dict_with_batch(images, labels)

        batch_info = model.evaluate(feed_dict)
        test_info = [a + b for a, b in zip(test_info, batch_info)]

    log = 'Test results: '
    log += 'loss={:.5f}, '.format(test_info[0] / num_steps)
    log += 'acc={:.5f}, '.format(test_info[1] / num_steps)

    print(log)

except KeyboardInterrupt:
    print()
    print('Test evaluation aborted.')
