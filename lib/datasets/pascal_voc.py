from __future__ import division
from __future__ import print_function

import os
from xml.dom.minidom import parse

import numpy as np
from skimage.io import imread
from skimage.transform import resize

from .dataset import Datasets
from .download import maybe_download_and_extract


URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'\
      'VOCtrainval_11-May-2012.tar'
CLASSES = [
    'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
    'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
    'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
]
WIDTH = 224
HEIGHT = 224
NUM_CHANNELS = 3


class PascalVOC(Datasets):
    def __init__(self, data_dir, val_size=1500):
        maybe_download_and_extract(URL, data_dir)

        data_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012')
        names = os.listdir(os.path.join(data_dir, 'Annotations'))
        names = [name.split('.')[0] for name in names]
        names = sorted(names)

        # PascalVOC didn't release the full test annotations yet, use the
        # validation set instead :(
        train = Dataset(names[val_size:], data_dir)
        val = Dataset(names[:val_size], data_dir)
        test = Dataset(names[:val_size], data_dir)

        super(PascalVOC, self).__init__(train, val, test)

    @property
    def classes(self):
        return CLASSES

    @property
    def width(self):
        return WIDTH

    @property
    def height(self):
        return HEIGHT

    @property
    def num_channels(self):
        return NUM_CHANNELS


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

        images = [self._read_image(name) for name in names]
        labels = np.stack([self._read_label(name) for name in names])

        return images, labels

    def _read_image(self, name):
        path = os.path.join(self._data_dir, 'JPEGImages',
                            '{}.jpg'.format(name))
        image = imread(path)
        # image = resize(image, (HEIGHT, WIDTH), mode='constant')
        # No need to cast to float when image is resized
        image = (1 / 255) * image.astype(np.float32)
        return image.astype(np.float32)

    def _read_label(self, name):
        path = os.path.join(self._data_dir, 'Annotations',
                            '{}.xml'.format(name))
        annotation = parse(path)

        label = np.zeros((len(CLASSES)), np.uint8)

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
