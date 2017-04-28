# from __future__ import division
# from __future__ import print_function

# import sys
# import os
# from six.moves import xrange
# from xml.dom.minidom import parse

# import numpy as np
# from skimage.io import imread

# from .dataset import Datasets, Dataset
# from .download import maybe_download_and_extract


# URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'\
#       'VOCtrainval_11-May-2012.tar'

# LABELS = [
#     'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
#     'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
#     'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
# ]


# class PascalVOC(Datasets):
#     def __init__(self, data_dir, validation_size=1500):
#         maybe_download_and_extract(URL, data_dir)
#         train_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012')

#         train_save_dir = os.path.join(data_dir, 'python', 'train')
#         train_images, train_labels = self._load_dataset(
#             train_dir, train_save_dir, 50, 'train and validation')

#         test_save_dir = os.path.join(data_dir, 'python', 'test')
#         test_images, test_labels = self._load_dataset(
#             data_dir, test_save_dir, 20, 'test')

#         train = Dataset(train_images[validation_size:],
#                         train_labels[validation_size:])
#         validation = Dataset(train_images[:validation_size],
#                              train_labels[:validation_size])
#         test = Dataset(test_images, test_labels)

#         super(PascalVOC, self).__init__(train, validation, test)

#     def label_name(self, label):
#         return [LABELS[i] for i in np.where(label == 1)[0]]

#     @property
#     def num_labels(self):
#         return len(LABELS)

#     def _load_dataset(self, data_dir, save_dir, num_files, dataset):
#         filenames = [
#             os.path.join(save_dir, '{}.p'.format(i + 1))
#             for i in xrange(num_files)
#         ]
#         labels_filename = os.path.join(save_dir, 'labels.p')

#         if os.path.exists(save_dir):
#             images = []
#             for i in xrange(num_files):
#                 images.append(pickle.load(open(filenames[i], 'rb')))
#             labels = pickle.load(open(labels_filename, 'rb'))
#         else:
#             images, labels = self._read_dataset(data_dir, dataset)

#             os.makedirs(save_dir)

#             start = 0
#             num_examples = len(images)
#             interval = num_examples // num_files + num_files
#             for i in xrange(num_files):
#                 end = start + interval
#                 pickle.dump(images[start:end], open(filenames[i], 'wb'))
#                 start = end

#                 sys.stdout.write('\r>> Saving {} dataset {:.2f}%'.format(
#                     dataset, 100 * (i + 1) / num_files))
#                 sys.stdout.flush()

#             pickle.dump(labels, open(labels_filename, 'wb'))

#         print()
#         return images, labels

#     def _read_dataset(self, data_dir, dataset):
#         filenames = os.listdir(os.path.join(data_dir, 'Annotations'))
#         filenames = [filename.split('.')[0] for filename in filenames]

#         num_examples = len(filenames)
#         images = []
#         labels = np.zeros((num_examples, self.num_labels), np.uint8)
#         for i in xrange(num_examples):
#             filename = filenames[i]
#             images.append(self._read_image(data_dir, filename))
#             labels[i] = self._read_label(data_dir, filename)

#             sys.stdout.write('\r>> Extracting {} dataset {:.2f}%'.format(
#                 dataset, 100 * (i + 1) / num_examples))
#             sys.stdout.flush()

#         print()
#         return images, labels

#     def _read_image(self, data_dir, filename):
#         image = imread(
#             os.path.join(data_dir, 'JPEGImages', '{}.jpg'.format(filename)))
#         return (1 / 255) * image.astype(np.float32)

#     def _read_label(self, data_dir, filename):
#         annotation = parse(
#             os.path.join(data_dir, 'Annotations', '{}.xml'.format(filename)))

#         label = np.zeros((self.num_labels), np.uint8)
#         for obj in annotation.getElementsByTagName('object'):
#             name = obj.getElementsByTagName('name')[0].firstChild.nodeValue
#             index = LABELS.index(name)
#             label[index] = 1
#         return label
