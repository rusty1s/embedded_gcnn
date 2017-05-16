from __future__ import division
from __future__ import print_function

import sys
import os
from xml.dom.minidom import parse

import numpy as np
from skimage.io import imread

from .dataset import Datasets, Dataset
from .download import maybe_download_and_extract


URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'\
      'VOCtrainval_11-May-2012.tar'


def _print_status(data_dir, percentage):
    sys.stdout.write('\r>> Reading {} {:.2f}%'.format(data_dir, percentage))
    sys.stdout.flush()


def _load_dataset(data_dir, classes, num_examples):
    names = os.listdir(os.path.join(data_dir, 'Annotations'))
    names = [name.split('.')[0] for name in names]
    names = sorted(names)

    images = []
    labels = []

    if num_examples is None:
        num_examples = len(names)
    else:
        num_examples = min(num_examples, len(names))
        names = names[:num_examples]

    for i, name in enumerate(names):
        label = _read_label(data_dir, name, classes)

        # Abort if no label found.
        if label.max() == 0:
            continue

        labels.append(label)
        images.append(_read_image(data_dir, name))

        if i % 20 == 0:
            _print_status(data_dir, 100 * i / num_examples)

    _print_status(data_dir, 100)
    print()

    return images, np.array(labels, dtype=np.uint8)


def _read_image(data_dir, name):
    path = os.path.join(data_dir, 'JPEGImages', '{}.jpg'.format(name))
    image = imread(path)
    image = (1 / 255) * image.astype(np.float32)
    return image


def _read_label(data_dir, name, classes):
    path = os.path.join(data_dir, 'Annotations', '{}.xml'.format(name))
    annotation = parse(path)

    label = np.zeros((len(classes)), np.uint8)

    for obj in annotation.getElementsByTagName('object'):
        # Pass difficult objects.
        difficult = obj.getElementsByTagName('difficult')
        if len(difficult) > 0:
            difficult = difficult[0].firstChild.nodeValue
            if difficult == '1':
                continue

        name = obj.getElementsByTagName('name')[0].firstChild.nodeValue

        try:
            index = classes.index(name)
            label[index] = 1
        except ValueError:
            pass

    return label


class PascalVOC(Datasets):
    def __init__(self,
                 data_dir,
                 num_examples=None,
                 val_size=1500,
                 classes=None):

        self._classes = classes

        maybe_download_and_extract(URL, data_dir)

        data_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012')
        images, labels = _load_dataset(data_dir, self.classes, num_examples)
        train = Dataset(images[val_size:], labels[val_size:])
        val = Dataset(images[:val_size], labels[:val_size])

        # PascalVOC doesn't have released the full test annotation, use
        # the validation set instead :(
        test = Dataset(images[:val_size], labels[:val_size])

        super(PascalVOC, self).__init__(train, val, test)

    @property
    def classes(self):
        if self._classes is None:
            return [
                'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
                'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike',
                'train', 'bottle', 'chair', 'diningtable', 'pottedplant',
                'sofa', 'tvmonitor'
            ]
        else:
            return self._classes
