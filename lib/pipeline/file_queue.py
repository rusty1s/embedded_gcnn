from __future__ import division

from threading import Thread, Event

try:
    from queue import Queue
except ImportError:
    from Queue import Queue


class FileQueue(object):
    def __init__(self, dataset, batch_size, capacity, shuffle=False):

        inputs = Queue(capacity // batch_size)
        stopper = Event()

        class ProducerThread(Thread):
            def run(self):
                while not stopper.isSet():
                    batch = dataset.next_batch(batch_size, shuffle)
                    inputs.put(batch)

        self._producer = ProducerThread()
        self._producer.start()
        self._inputs = inputs
        self._stopper = stopper

    def dequeue(self):
        batch = self._inputs.get()
        self._inputs.task_done()
        return batch

    def close(self):
        self._stopper.set()

        # Delete all batches in queue.
        while not self._inputs.empty():
            self._inputs.get()
            self._inputs.task_done()
        self._inputs.join()

        # Shut down producer thread.
        self._producer.join()
