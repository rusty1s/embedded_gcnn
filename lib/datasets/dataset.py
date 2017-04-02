from __future__ import print_function
from __future__ import division

import os
import sys
from six.moves import xrange

import numpy as np

try:
    import cPickle as pickle
except:
    import pickle


class Datasets(object):
    def __init__(self,
                 train,
                 validation,
                 test,
                 preprocess=False,
                 data_dir=None):

        if preprocess and data_dir is not None:
            # Create directory.
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            # Load or create/save train data.
            train_file = os.path.join(data_dir, 'train.p')
            if os.path.exists(train_file):
                train._data = pickle.load(open(train_file, 'rb'))
            else:
                train._data = self._preprocess_all(
                    train._data, self._train_preprocess, 'train')
                print('Resaving train dataset...')
                pickle.dump(train._data, open(train_file, 'wb'))

            # Load or create/save validation data.
            validation_file = os.path.join(data_dir, 'validation.p')
            if os.path.exists(validation_file):
                validation._data = pickle.load(open(validation_file, 'rb'))
            else:
                validation._data = self._preprocess_all(
                    validation._data, self._eval_preprocess, 'validation')
                print('Resaving validation dataset...')
                pickle.dump(validation._data, open(validation_file, 'wb'))

            # Load or create/save test data.
            test_file = os.path.join(data_dir, 'test.p')
            if os.path.exists(test_file):
                test._data = pickle.load(open(test_file, 'rb'))
            else:
                test._data = self._preprocess_all(
                    test._data, self._eval_preprocess, 'test')
                print('Resaving test dataset...')
                pickle.dump(test._data, open(test_file, 'wb'))

        elif preprocess:
            # Preprocess every time dataset is initialized.
            train._data = self._preprocess_all(train._data,
                                               self._train_preprocess, 'train')
            validation._data = self._preprocess_all(
                validation._data, self._eval_preprocess, 'validation')
            test._data = self._preprocess_all(test._data,
                                              self._eval_preprocess, 'test')

        self.train = train
        self.validation = validation
        self.test = test

        # Set postprocessing function for each set.
        self.train._postprocess = self._train_postprocess
        self.validation._postprocess = self._eval_postprocess
        self.test._postprocess = self._eval_postprocess

    def label_name(self, label):
        raise NotImplementedError

    @property
    def num_labels(self):
        raise NotImplementedError

    def _preprocess_all(self, data, preprocess, dataset):
        size = len(data) if isinstance(data, list) else data.shape[0]

        def _preprocess_single_data(data, index):
            sys.stdout.write('\r>> Preprocessing {} dataset {:.2f}%'.format(
                dataset, 100 * (index + 1) / size))
            sys.stdout.flush()
            return preprocess(data)

        data = [_preprocess_single_data(data[i], i) for i in xrange(size)]
        print()
        return data

    def _preprocess(self, data):
        return data

    def _train_preprocess(self, data):
        return self._preprocess(data)

    def _eval_preprocess(self, data):
        return self._preprocess(data)

    def _postprocess(self, data_batch):
        return data_batch

    def _train_postprocess(self, data_batch):
        return self._postprocess(data_batch)

    def _eval_postprocess(self, data_batch):
        return self._postprocess(data_batch)


class Dataset(object):
    def __init__(self, data, labels):
        self.epochs_completed = 0
        self._data = data  # Numpy array or a python list.
        self._labels = labels  # Numpy array.
        self._postprocess = None
        self._index_in_epoch = 0

    @property
    def num_examples(self):
        return self._labels.shape[0]

    def _random_shuffle_examples(self):
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        if isinstance(self._data, list):
            self._data = [self._data[i] for i in perm]
        else:
            self._data = self._data[perm]
        self._labels = self._labels[perm]

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
            data_rest = self._data[start:self.num_examples]
            labels_rest = self._labels[start:self.num_examples]

            # Shuffle the data.
            if shuffle:
                self._random_shuffle_examples()

            # Start next epoch.
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new = self._data[start:end]
            labels_new = self._labels[start:end]

            labels_batch = np.concatenate((labels_rest, labels_new), axis=0)
            # Check if we look at python lists or numpy arrays.
            if isinstance(self._data, list):
                data_batch = data_rest + data_new
            else:
                data_batch = np.concatenate((data_rest, data_new), axis=0)
        else:
            # Just slice the data.
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            data_batch = self._data[start:end]
            labels_batch = self._labels[start:end]

        # Postprocess data.
        if self._postprocess is not None:
            data_batch = self._postprocess(data_batch)

        return data_batch, labels_batch
