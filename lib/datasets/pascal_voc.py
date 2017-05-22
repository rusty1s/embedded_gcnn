from __future__ import division
from __future__ import print_function

import os
from xml.dom.minidom import parse

import numpy as np
from skimage.io import imread

from .dataset import Datasets
from .download import maybe_download_and_extract


URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'\
      'VOCtrainval_11-May-2012.tar'
CLASSES = [
    'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
    'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
    'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
]


def _read_image(name, data_dir):
    path = os.path.join(data_dir, 'JPEGImages', '{}.jpg'.format(name))
    image = imread(path)
    image = (1 / 255) * image.astype(np.float32)
    return image


def _read_label(name, data_dir):
    path = os.path.join(data_dir, 'Annotations', '{}.xml'.format(name))
    annotation = parse(path)

    label = np.zeros((len(CLASSES)), np.uint8)

    for obj in annotation.getElementsByTagName('object'):
        # Pass difficult objects.
        difficult = obj.getElementsByTagName('difficult')
        if len(difficult) > 0:
            difficult = difficult[0].firstChild.nodeValue
            if difficult == '1':
                continue

        name = obj.getElementsByTagName('name')[0].firstChild.nodeValue

        try:
            index = CLASSES.index(name)
            label[index] = 1
        except ValueError:
            pass

    return label


class PascalVOC(Datasets):
    def __init__(self, data_dir, val_size=1500):
        maybe_download_and_extract(URL, data_dir)

        data_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012')
        names = os.listdir(os.path.join(data_dir, 'Annotations'))
        names = [name.split('.')[0] for name in names]
        names = sorted(names)

        # # PascalVOC doesn't have released the full test annotation, use
        # # the validation set instead :(
        train = Dataset(names[val_size:], self.classes, data_dir)
        val = Dataset(names[:val_size], self.classes, data_dir)
        test = Dataset(names[:val_size], self.classes, data_dir)

        super(PascalVOC, self).__init__(train, val, test)

    @property
    def classes(self):
        return CLASSES


class Dataset(object):
    def __init__(self, names, classes, data_dir):
        self.epochs_completed = 0
        self._data_dir = data_dir
        self._names = names
        self._classes = classes
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

        images = [_read_image(name, self._data_dir) for name in names]
        labels = np.array(
            [_read_label(name, self._data_dir) for name in names], np.uint8)

        return images, labels
