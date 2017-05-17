import tensorflow as tf

from .metrics import top_accuracy, threshold_accuracy


class MetricsTest(tf.test.TestCase):
    def test_top_accuracy(self):
        outputs = [[8, 5, 3, 9], [3, 4, 6, 4]]
        # [[0, 0, 0, 1], [0, 0, 1, 0]]
        outputs = tf.constant(outputs, tf.float32)

        # None right.
        labels = [[0, 1, 0, 0], [0, 1, 0, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 0)

        # All right.
        labels = [[0, 0, 0, 1], [0, 0, 1, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 1)

        # 1 out of 2 right.
        labels = [[1, 0, 0, 0], [0, 0, 1, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 0.5)

    def test_top_accuracy_multilabel(self):
        outputs = [[8, 5, 3, 9], [3, 4, 6, 4]]
        # [[0, 0, 0, 1], [0, 0, 1, 0]]
        outputs = tf.constant(outputs, tf.float32)

        # None right.
        labels = [[0, 1, 1, 0], [0, 1, 0, 1]]
        labels = tf.constant(labels, tf.int32)
        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 0)

        # All right.
        labels = [[0, 1, 0, 1], [1, 0, 1, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 1)

        # 1 out of 2 right.
        labels = [[1, 1, 0, 0], [0, 1, 1, 1]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 0.5)

    def test_threshold_accuracy(self):
        outputs = [[-2, -1, 0, 1, 2], [1, 2, 0, -1, -2]]
        # [[0, 0, 0, 1, 1], [1, 1, 0, 0, 0]]
        outputs = tf.constant(outputs, tf.float32)

        # None right.
        labels = [[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(threshold_accuracy(outputs, labels).eval(), 0.0)

        # All right.
        labels = [[0, 0, 0, 1, 1], [1, 1, 0, 0, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(threshold_accuracy(outputs, labels).eval(), 1.0)

        # 6 out of 10 right.
        labels = [[1, 0, 0, 1, 0], [1, 0, 1, 0, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertAlmostEqual(
                threshold_accuracy(outputs, labels).eval(), 0.6)
