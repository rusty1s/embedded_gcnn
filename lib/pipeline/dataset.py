from __future__ import print_function
from __future__ import division

import os
import sys
from six.moves import xrange

import numpy as np


def _print_status(data_dir, percentage):
    sys.stdout.write(
        '\r>> Preprocessing to {} {:.2f}%'.format(data_dir, percentage))
    sys.stdout.flush()


def _save(data_dir, name, features, adjs_dist, adjs_rad, label):
    data = (features, adjs_dist, adjs_rad, label)
    np.save(os.path.join(data_dir, name), data)


def _load(data_dir, names):
    batch = []
    for name in names:
        path = os.path.join(data_dir, name)
        batch.append(np.load(path))
    return batch


class PreprocessedDataset(object):
    def __init__(self, data_dir, dataset, preprocess_algorithm):
        self._data_dir = data_dir
        self.epochs_completed = 0
        self._index_in_epoch = 0

        if os.path.exists(data_dir):
            self._names = os.listdir(data_dir)
        else:
            os.makedirs(data_dir)
            num_count = len(str(dataset.num_examples))
            self._names = [
                '{}.npy'.format(str(i).zfill(num_count))
                for i in xrange(dataset.num_examples)
            ]

            num_left = dataset.num_examples
            batch_size = 25

            j = 0
            while num_left > 0:
                min_batch = min(batch_size, num_left)
                images, labels = dataset.next_batch(min_batch, shuffle=False)
                num_left -= min_batch

                for i in xrange(labels.shape[0]):
                    f, adjs_dist, adjs_rad = preprocess_algorithm(images[i])
                    _save(data_dir, self._names[j], f, adjs_dist, adjs_rad,
                          labels[i])
                    j += 1
                _print_status(data_dir,
                              100 * (1 - num_left / dataset.num_examples))

            _print_status(data_dir, 100)
            print()

    @property
    def num_examples(self):
        return len(self._names)

    def _random_shuffle_examples(self):
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self._names = [self._names[i] for i in perm]

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch

        # Shuffle for the first epoch.
        if self.epochs_completed == 0 and start == 0 and shuffle:
            self._random_shuffle_examples()

        if start + batch_size > self.num_examples:
            # Finished epoch.
            self.epochs_completed += 1

            # Get the rest examples in this epoch.
            rest_num_examples = self.num_examples - start
            names_rest = self._names[start:self.num_examples]

            # Shuffle the examples.
            if shuffle:
                self._random_shuffle_examples()

            # Start next epoch.
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            names = names_rest + self._names[start:end]
        else:
            # Just slice the examples.
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            names = self._names[start:end]

        return _load(self._data_dir, names)
