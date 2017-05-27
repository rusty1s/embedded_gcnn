from __future__ import print_function
from __future__ import division

from six.moves import xrange

import os
import time

from ..datasets import PreprocessQueue
from .placeholder import feed_dict_with_batch
from ..pipeline.dataset import PreprocessedDataset
from ..pipeline.file_queue import FileQueue
from ..pipeline.augment import augment_batch


def train(model,
          data,
          preprocess_algorithm,
          batch_size,
          dropout,
          augment,
          max_steps,
          preprocess_first=None,
          display_step=10,
          save_step=250):

    capacity = 10 * batch_size

    if preprocess_first is not None:
        data_dir = preprocess_first
        data.train = PreprocessedDataset(
            os.path.join(data_dir, 'train'), data.train, preprocess_algorithm)
        data.val = PreprocessedDataset(
            os.path.join(data_dir, 'val'), data.val, preprocess_algorithm)
        data.test = PreprocessedDataset(
            os.path.join(data_dir, 'test'), data.test, preprocess_algorithm)

        train_queue = FileQueue(data.train, batch_size, capacity, shuffle=True)
        val_queue = FileQueue(data.val, batch_size, capacity, shuffle=True)
    else:
        train_queue = PreprocessQueue(
            data.train,
            preprocess_algorithm,
            batch_size,
            capacity,
            shuffle=True)

        val_queue = PreprocessQueue(
            data.val, preprocess_algorithm, batch_size, capacity, shuffle=True)

    model.build()
    global_step = model.initialize()

    try:
        for step in xrange(global_step, max_steps):
            t_pre = time.process_time()
            batch = train_queue.dequeue()
            batch = augment_batch(batch) if augment else batch
            feed_dict = feed_dict_with_batch(model.placeholders, batch,
                                             dropout)
            t_pre = time.process_time() - t_pre

            t_train = model.train(feed_dict, step)

            if step % display_step == 0:
                # Evaluate on training and validation set with zero dropout.
                feed_dict.update({model.placeholders['dropout']: 0})
                batch = val_queue.dequeue()
                val_feed_dict = feed_dict_with_batch(model.placeholders, batch)

                train_info = model.evaluate(feed_dict, step, 'train')
                val_info = model.evaluate(val_feed_dict, step, 'val')

                log = 'step={}, '.format(step)
                log += 'time={:.2f}s + {:.2f}s, '.format(t_pre, t_train)
                log += 'train_loss={:.5f}, '.format(train_info[0])
                if not model.isMultilabel:
                    log += 'train_acc={:.5f}, '.format(train_info[1])
                else:
                    log += 'train_top_acc={:.5f}, '.format(train_info[1])
                    log += 'train_threshold_acc={:.5f}, '.format(train_info[2])
                log += 'val_loss={:.5f}, '.format(val_info[0])
                if not model.isMultilabel:
                    log += 'val_acc={:.5f}'.format(val_info[1])
                else:
                    log += 'val_top_acc={:.5f}, '.format(val_info[1])
                    log += 'val_threshold_acc={:.5f}'.format(val_info[2])

                print(log)

            if step % save_step == 0:
                model.save()

    except KeyboardInterrupt:
        print()

    finally:
        train_queue.close()
        val_queue.close()

    print('Optimization finished!')
    print('Evaluate on test set. This can take a few minutes.')

    try:
        if preprocess_first is not None:
            test_queue = FileQueue(
                data.test, batch_size, capacity, shuffle=False)
        else:
            test_queue = PreprocessQueue(
                data.test,
                preprocess_algorithm,
                batch_size,
                capacity,
                shuffle=False)

        num_steps = data.test.num_examples // batch_size
        test_info = [0, 0, 0] if model.isMultilabel else [0, 0]

        for i in xrange(num_steps):
            batch = test_queue.dequeue()
            feed_dict = feed_dict_with_batch(model.placeholders, batch)

            batch_info = model.evaluate(feed_dict)
            test_info = [a + b for a, b in zip(test_info, batch_info)]

        log = 'Test results: '
        log += 'loss={:.5f}, '.format(test_info[0] / num_steps)
        if not model.isMultilabel:
            log += 'acc={:.5f}'.format(test_info[1] / num_steps)
        else:
            log += 'top_acc={:.5f}, '.format(test_info[1] / num_steps)
            log += 'threshold_acc={:.5f}'.format(test_info[2] / num_steps)

        print(log)

    except KeyboardInterrupt:
        print()
        print('Test evaluation aborted.')

    finally:
        test_queue.close()
