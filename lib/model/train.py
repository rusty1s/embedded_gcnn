from __future__ import print_function
from __future__ import division

from six.moves import xrange

import time

from ..datasets import PreprocessQueue
from .placeholder import feed_dict_with_batch
from ..pipeline.dataset import PreprocessedDataset


def train(model,
          data,
          preprocess_algorithm,
          batch_size,
          dropout,
          max_steps,
          preprocess_first=False,
          display_step=10,
          save_step=250):

    global_step = model.initialize()

    capacity = 10 * batch_size

    if preprocess_first:
        data.train = PreprocessedDataset(data.train, preprocess_algorithm)
        data.val = PreprocessedDataset(data.val, preprocess_algorithm)
        data.test = PreprocessedDataset(data.test, preprocess_algorithm)

    try:
        if not preprocess_first:
            train_queue = PreprocessQueue(
                data.train,
                preprocess_algorithm,
                batch_size,
                capacity,
                shuffle=True)

            val_queue = PreprocessQueue(
                data.val,
                preprocess_algorithm,
                batch_size,
                capacity,
                shuffle=True)

        for step in xrange(global_step, max_steps):
            t_preprocess = time.process_time()

            if not preprocess_first:
                batch = train_queue.dequeue()
            else:
                batch = data.train.next_batch(batch_size, shuffle=True)

            feed_dict = feed_dict_with_batch(model.placeholders, batch,
                                             dropout)
            t_preprocess = time.process_time() - t_preprocess

            t_train = model.train(feed_dict, step)

            if step % display_step == 0:
                # Evaluate on training and validation set with zero dropout.
                feed_dict.update({model.placeholders['dropout']: 0})
                train_loss, train_acc = model.evaluate(feed_dict)

                if not preprocess_first:
                    batch = val_queue.dequeue()
                else:
                    batch = data.val.next_batch(batch_size, shuffle=True)

                feed_dict = feed_dict_with_batch(model.placeholders, batch)
                val_loss, val_acc = model.evaluate(feed_dict)

                print(', '.join([
                    'step: {}'.format(step),
                    'time={:.2f}s + {:.2f}s'.format(t_preprocess, t_train),
                    'train_loss={:.5f}'.format(train_loss),
                    'train_acc={:.5f}'.format(train_acc),
                    'val_loss={:.5f}'.format(val_loss),
                    'val_acc={:.5f}'.format(val_acc),
                ]))

            if step % save_step == 0:
                model.save()

    except KeyboardInterrupt:
        print()

    finally:
        if not preprocess_first:
            train_queue.close()
            val_queue.close()

    print('Optimization finished!')
    print('Evaluate on test set. This can take a few minutes.')

    try:
        if not preprocess_first:
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
            if not preprocess_first:
                batch = test_queue.dequeue()
            else:
                batch = data.test.next_batch(batch_size, shuffle=False)

            feed_dict = feed_dict_with_batch(model.placeholders, batch)
            batch_loss, batch_acc = model.evaluate(feed_dict)
            loss += batch_loss
            acc += batch_acc

        loss /= num_steps
        acc /= num_steps

        print('Test results: cost={:.5f}, acc={:.5f}'.format(loss, acc))

    except KeyboardInterrupt:
        print()
        print('Test evaluation aborted.')

    finally:
        if not preprocess_first:
            test_queue.close()
