from __future__ import print_function
from __future__ import division

from six.moves import xrange

import time

from ..datasets import PreprocessQueue
from .placeholder import feed_dict_with_batch


def train(model,
          data,
          preprocess_algorithm,
          batch_size,
          dropout,
          max_steps,
          display_step=10):

    global_step = model.initialize()

    capacity = 10 * batch_size

    try:
        train_queue = PreprocessQueue(
            data.train,
            preprocess_algorithm,
            batch_size,
            capacity,
            shuffle=True)

        val_queue = PreprocessQueue(
            data.validation,
            preprocess_algorithm,
            batch_size,
            capacity,
            shuffle=True)

        for step in xrange(global_step, max_steps):
            t_preprocess = time.process_time()
            batch = train_queue.dequeue()
            feed_dict = feed_dict_with_batch(model.placeholders, batch,
                                             dropout)
            t_preprocess = time.process_time() - t_preprocess

            t_train = model.train(feed_dict, step)

            if step % display_step == 0:
                # Evaluate on training and validation set with zero dropout.
                feed_dict.update({model.placeholders['dropout']: 0})
                train_loss, train_acc = model.evaluate(feed_dict)

                batch = val_queue.dequeue()
                feed_dict = feed_dict_with_batch(model.placeholders, batch)
                val_loss, val_acc = model.evaluate(feed_dict)

                print(', '.join([
                    'step: {}'.format(step),
                    'time={:.2f}+{:.2f}s'.format(t_preprocess, t_train),
                    'train_loss={:.5f}'.format(train_loss),
                    'train_acc={:.5f}'.format(train_acc),
                    'val_loss={:.5f}'.format(val_loss),
                    'val_acc={:.5f}'.format(val_acc),
                ]))

    except KeyboardInterrupt:
        pass

    finally:
        train_queue.close()
        val_queue.close()

    print('Optimization finished!')
    print('Evaluate on test set. This can take a few minutes.')

    try:
        test_queue = PreprocessQueue(
            data.test,
            preprocess_algorithm,
            batch_size,
            capacity,
            shuffle=False)

        num_steps = data.test.num_examples // batch_size
        loss = 0
        acc = 0

        for i in xrange(num_steps):
            batch = test_queue.dequeue()
            feed_dict = feed_dict_with_batch(model.placeholders, batch)
            batch_loss, batch_acc = model.evaluate(feed_dict)
            loss += batch_loss
            acc += batch_acc

        loss /= num_steps
        acc /= num_steps

        print('Test results: cost={:.5f}, acc={:.5f}'.format(loss, acc))

    except KeyboardInterrupt:
        print('Test evaluation aborted.')

    finally:
        test_queue.close()
