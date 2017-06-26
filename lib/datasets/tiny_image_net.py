import os
from six.moves import xrange

import numpy as np
from skimage.io import imread
from skimage.color import gray2rgb

from .dataset import Datasets
from .download import maybe_download_and_extract

URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
NUM_CLASSES = 200


class TinyImageNet(Datasets):
    def __init__(self, data_dir):
        data_dir = maybe_download_and_extract(URL, data_dir)

        with open(os.path.join(data_dir, 'words.txt')) as f:
            class_dict = dict(line.strip().split('\t') for line in f)

        with open(os.path.join(data_dir, 'wnids.txt')) as f:
            ids = [line.strip() for line in f]
            self._classes = [class_dict[idx] for idx in ids]

            train_names = []
            for label, idx in enumerate(ids):
                names = [(os.path.join('train', idx, 'images',
                                       '{}_{}.JPEG'.format(idx, i)), label)
                         for i in xrange(500)]
                train_names.extend(names)

            val_annotations_filename = os.path.join(data_dir, 'val',
                                                    'val_annotations.txt')
            with open(val_annotations_filename) as f:
                val_names = [(os.path.join('val', 'images',
                                           line.strip().split('\t')[0]),
                              ids.index(line.strip().split('\t')[1]))
                             for line in f]

            train = Dataset(train_names, data_dir)
            val = Dataset(val_names, data_dir)
            test = Dataset(val_names, data_dir)

            super(TinyImageNet, self).__init__(train, val, test)

    @property
    def classes(self):
        return self._classes

    @property
    def width(self):
        return 64

    @property
    def height(self):
        return 64

    @property
    def num_channels(self):
        return 3


class Dataset(object):
    def __init__(self, names, data_dir):
        self.epochs_completed = 0

        self._data_dir = data_dir
        self._names = names
        self._index_in_epoch = 0

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

        images = np.stack([self._read_image(name[0]) for name in names])
        labels = np.stack([self._convert_label(name[1]) for name in names])

        return images, labels

    def _read_image(self, name):
        image = imread(os.path.join(self._data_dir, name))

        if len(image.shape) == 2:
            image = gray2rgb(image)

        image = (1 / 255) * image.astype(np.float32)
        return image.astype(np.float32)

    def _convert_label(self, label):
        label_one_hot = np.zeros(NUM_CLASSES, np.uint8)
        label_one_hot[label] = 1
        return label_one_hot
