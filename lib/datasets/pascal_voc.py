from __future__ import division

import os
from math import ceil, floor
from xml.dom.minidom import parse

import numpy as np
from skimage.io import imread
from skimage.transform import rescale

from .dataset import Datasets
from .download import maybe_download_and_extract


URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'\
      'VOCtrainval_11-May-2012.tar'
CLASSES = [
    'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
    'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
    'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
]


class PascalVOC(Datasets):
    def __init__(self, data_dir, val_size=1500, fixed_size=None):
        self._fixed_size = fixed_size

        maybe_download_and_extract(URL, data_dir)

        data_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012')
        names = os.listdir(os.path.join(data_dir, 'Annotations'))
        names = [name.split('.')[0] for name in names]
        names = sorted(names)

        # PascalVOC didn't release the full test annotations yet, use the
        # validation set instead :(
        train = Dataset(names[val_size:], data_dir, fixed_size)
        val = Dataset(names[:val_size], data_dir, fixed_size)
        test = Dataset(names[:val_size], data_dir, fixed_size)

        super(PascalVOC, self).__init__(train, val, test)

    @property
    def classes(self):
        return CLASSES

    @property
    def width(self):
        return self._fixed_size

    @property
    def height(self):
        return self._fixed_size

    @property
    def num_channels(self):
        return 3


class Dataset(object):
    def __init__(self, names, data_dir, fixed_size=None):
        self.epochs_completed = 0

        self._data_dir = data_dir
        self._fixed_size = fixed_size
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

        if self._fixed_size is None:
            images = [self._read_image(name) for name in names]
        else:
            images = np.stack([self._read_image(name) for name in names])
        labels = np.stack([self._read_label(name) for name in names])

        return images, labels

    def _read_image(self, name):
        path = os.path.join(self._data_dir, 'JPEGImages',
                            '{}.jpg'.format(name))
        image = imread(path)

        if self._fixed_size is None:
            image = (1 / 255) * image.astype(np.float32)
            return image.astype(np.float32)
        else:
            height, width, _ = image.shape

            scale_y = self._fixed_size / height
            scale_x = self._fixed_size / width
            scale = min(scale_y, scale_x)

            image = rescale(image, (scale, scale), mode='constant')

            pad_y = self._fixed_size - image.shape[0]
            pad_x = self._fixed_size - image.shape[1]

            image = np.pad(image, ((ceil(pad_y / 2), floor(pad_y / 2)), (ceil(
                pad_x / 2), floor(pad_x / 2)), (0, 0)), 'constant')
            return image

    def _read_label(self, name):
        path = os.path.join(self._data_dir, 'Annotations',
                            '{}.xml'.format(name))
        annotation = parse(path)

        label = np.zeros((len(CLASSES)), np.uint8)

        # Get the label to the greatest bounding box
        max_area = 0
        max_name = ''
        for obj in annotation.getElementsByTagName('object'):
            name = obj.getElementsByTagName('name')[0].firstChild.nodeValue
            bbox = obj.getElementsByTagName('bndbox')[0]
            xmin = bbox.getElementsByTagName('xmin')[0].firstChild.nodeValue
            xmax = bbox.getElementsByTagName('xmax')[0].firstChild.nodeValue
            ymin = bbox.getElementsByTagName('ymin')[0].firstChild.nodeValue
            ymax = bbox.getElementsByTagName('ymax')[0].firstChild.nodeValue
            area = (float(xmax) - float(xmin)) * (float(ymax) - float(ymin))
            if area > max_area:
                max_area = area
                max_name = name

        label[CLASSES.index(max_name)] = 1

        return label
