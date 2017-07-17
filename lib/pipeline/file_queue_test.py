from unittest import TestCase
import time

from .file_queue import FileQueue
from ..datasets import MNIST

data = MNIST('data/mnist')


class QueueTest(TestCase):
    def test_dequeue(self):
        queue = FileQueue(data.train, batch_size=2, capacity=8, shuffle=False)

        batch = queue.dequeue()

        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].shape, (2, 28, 28, 1))
        self.assertEqual(batch[1].shape, (2, 10))

        # Ensure items in inputs queues before closing.
        time.sleep(5)
        queue.close()
