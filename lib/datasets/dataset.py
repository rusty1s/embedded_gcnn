import numpy as np


class Datasets(object):
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        return len(self.classes)

    def classnames(self, label):
        idx = np.where(label == 1)[0]
        return [self.classes[i] for i in idx]


class Dataset(object):
    def __init__(self, images, labels):
        self.epochs_completed = 0
        self._images = images
        self._labels = labels
        self._index_in_epoch = 0

    @property
    def num_examples(self):
        return self._labels.shape[0]

    def _random_shuffle_examples(self):
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
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
            images_rest = self._images[start:self.num_examples]
            labels_rest = self._labels[start:self.num_examples]

            # Shuffle the examples.
            if shuffle:
                self._random_shuffle_examples()

            # Start next epoch.
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new = self._images[start:end]
            labels_new = self._labels[start:end]

            labels = np.concatenate((labels_rest, labels_new), axis=0)
            images = np.concatenate((images_rest, images_new), axis=0)
        else:
            # Just slice the examples.
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            images = self._images[start:end]
            labels = self._labels[start:end]

        return images, labels
