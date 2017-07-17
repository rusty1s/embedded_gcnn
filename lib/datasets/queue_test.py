from unittest import TestCase
import time

from .queue import PreprocessQueue
from .mnist import MNIST
from ..pipeline import preprocess_pipeline_fixed
from ..segmentation import slic_fixed, extract_features_fixed

data = MNIST('data/mnist')
segmentation_algorithm = slic_fixed(
    num_segments=100, compactness=5, max_iterations=10, sigma=0)
feature_extraction_algorithm = extract_features_fixed([0, 1, 2])
preprocess_algorithm = preprocess_pipeline_fixed(
    segmentation_algorithm, feature_extraction_algorithm, 2)


class QueueTest(TestCase):
    def test_dequeue(self):
        queue = PreprocessQueue(
            data.train,
            preprocess_algorithm,
            augment=True,
            batch_size=2,
            capacity=8,
            shuffle=False)

        batch = queue.dequeue()
        example = batch[0]
        features, adjs_dist, adjs_rad, label = example

        self.assertEqual(len(batch), 2)
        self.assertEqual(len(example), 4)
        self.assertEqual(features.shape[1], 4)
        self.assertEqual(len(adjs_dist), 3)
        self.assertEqual(len(adjs_rad), 3)
        self.assertEqual(label.shape[0], 10)

        # Ensure items in both queues before closing.
        time.sleep(5)
        queue.close()
