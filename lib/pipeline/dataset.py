from __future__ import print_function
from __future__ import division

import sys
from six.moves import xrange

import numpy as np

from .preprocess import preprocess_pipeline


def _print_status(percentage):
    sys.stdout.write('\r>> Preprocessing {:.2f}%'.format(percentage))
    sys.stdout.flush()


class PreprocessedDataset(object):
    def __init__(self,
                 dataset,
                 segmentation_algorithm,
                 feature_extraction_algorithm,
                 levels,
                 filter_algorithm=None,
                 scale_invariance=False,
                 stddev=1):

        self.epochs_completed = 0
        self._index_in_epoch = 0

        images, labels = dataset.next_batch(
            dataset.num_examples, shuffle=False)

        self._data = []
        for i in xrange(dataset.num_examples):
            features, adjs_dist, adjs_rad = preprocess_pipeline(
                images[i], segmentation_algorithm,
                feature_extraction_algorithm, levels, filter_algorithm,
                scale_invariance, stddev)

            self._data.append((features, adjs_dist, adjs_rad, labels[i]))

            if i % 20 == 0:
                _print_status(100 * i / dataset.num_examples)

        _print_status(100)
        print()

    @property
    def num_examples(self):
        return len(self._data)

    def _random_shuffle_examples(self):
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self._data = [self._data[i] for i in perm]

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
            batch_rest = self._data[start:self.num_examples]

            # Shuffle the examples.
            if shuffle:
                self._random_shuffle_examples()

            # Start next epoch.
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            batch_new = self._data[start:end]
            batch = batch_rest + batch_new
        else:
            # Just slice the examples.
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            batch = self._data[start:end]

        return batch
