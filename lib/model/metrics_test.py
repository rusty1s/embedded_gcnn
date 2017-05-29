import tensorflow as tf

from .metrics import accuracy, precision_recall


class MetricsTest(tf.test.TestCase):
    def test_accuracy(self):
        outputs = [[8, 5, 3, 9], [3, 4, 6, 4]]
        # [[0, 0, 0, 1], [0, 0, 1, 0]]
        outputs = tf.constant(outputs, tf.float32)

        # None right.
        labels = [[0, 1, 0, 0], [0, 1, 0, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(accuracy(outputs, labels).eval(), 0)

        # All right.
        labels = [[0, 0, 0, 1], [0, 0, 1, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(accuracy(outputs, labels).eval(), 1)

        # 1 out of 2 right.
        labels = [[1, 0, 0, 0], [0, 0, 1, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(accuracy(outputs, labels).eval(), 0.5)

    def test_accuracy_multilabel(self):
        outputs = [[8, 5, 3, 9], [3, 4, 6, 4]]
        # [[0, 0, 0, 1], [0, 0, 1, 0]]
        outputs = tf.constant(outputs, tf.float32)

        # None right.
        labels = [[0, 1, 1, 0], [0, 1, 0, 1]]
        labels = tf.constant(labels, tf.int32)
        with self.test_session():
            self.assertEqual(accuracy(outputs, labels).eval(), 0)

        # All right.
        labels = [[0, 1, 0, 1], [1, 0, 1, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(accuracy(outputs, labels).eval(), 1)

        # 1 out of 2 right.
        labels = [[1, 1, 0, 0], [0, 1, 1, 1]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(accuracy(outputs, labels).eval(), 0.5)

    def test_precision_recall(self):
        outputs = [[-2, -1, 0, 1, 2], [1, 2, 0, -1, -2]]
        # [[0, 0, 0, 1, 1], [1, 1, 0, 0, 0]]
        outputs = tf.constant(outputs, tf.float32)

        # None right.
        labels = [[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(precision_recall(outputs, labels).eval(), 0.0)

        # All right.
        labels = [[0, 0, 0, 1, 1], [1, 1, 0, 0, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(precision_recall(outputs, labels).eval(), 1.0)

        # 6 out of 10 right.
        labels = [[1, 0, 0, 1, 0], [1, 0, 1, 0, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertAlmostEqual(
                precision_recall(outputs, labels).eval(), 0.6)
