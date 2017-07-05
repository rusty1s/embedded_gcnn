from threading import Thread, Event
from six.moves import xrange

from .augment import (random_flip_left_right_image, random_brightness,
                      random_contrast)

try:
    from queue import Queue
except ImportError:
    from Queue import Queue


class PreprocessQueue(object):
    def __init__(self,
                 dataset,
                 preprocess_algorithm,
                 augment,
                 batch_size,
                 capacity,
                 shuffle=False,
                 num_threads=1):

        inputs = Queue(capacity)
        outputs = Queue(capacity)
        stopper = Event()

        class ProducerThread(Thread):
            def run(self):
                while not stopper.isSet():
                    images, labels = dataset.next_batch(batch_size, shuffle)
                    for i in xrange(batch_size):
                        inputs.put((images[i], labels[i]))

        class ConsumerThread(Thread):
            def run(self):
                while not stopper.isSet():
                    image, label = inputs.get()
                    inputs.task_done()

                    if augment:
                        image = random_flip_left_right_image(image)
                        image = random_brightness(image, max_delta=0.3)
                        image = random_contrast(image, max_delta=0.3)

                    data = preprocess_algorithm(image)
                    data += (label, )

                    outputs.put(data)

        self._threads = [ProducerThread()]

        for i in xrange(num_threads):
            self._threads.append(ConsumerThread())
        for t in self._threads:
            t.start()

        self._stopper = stopper
        self._inputs = inputs
        self._outputs = outputs
        self._batch_size = batch_size

    def dequeue(self):
        batch = []

        for i in xrange(self._batch_size):
            data = self._outputs.get()
            self._outputs.task_done()
            batch.append(data)

        return batch

    def close(self):
        self._stopper.set()

        # Delete all items in both queues.
        while not self._inputs.empty():
            self._inputs.get()
            self._inputs.task_done()
        self._inputs.join()

        while not self._outputs.empty():
            self._outputs.get()
            self._outputs.task_done()
        self._outputs.join()

        # Shut down all threads.
        for t in self._threads:
            t.join()
