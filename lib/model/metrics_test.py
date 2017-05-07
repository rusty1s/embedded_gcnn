import tensorflow as tf

from .metrics import top_accuracy


class MetricsTest(tf.test.TestCase):
    def test_top_accuracy(self):
        outputs = [[0.8, 0.5, 0.3, 0.9], [0.3, 0.4, 0.6, 0.4]]
        outputs = tf.constant(outputs, tf.float32)

        labels = [[1, 0, 0, 0], [0, 0, 1, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 0.5)

        labels = [[0, 0, 0, 1], [0, 0, 1, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 1)

        labels = [[0, 1, 0, 0], [0, 1, 0, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 0)

    def test_top_accuracy_multilabel(self):
        outputs = [[0.8, 0.5, 0.3, 0.9], [0.3, 0.4, 0.6, 0.4]]
        outputs = tf.constant(outputs, tf.float32)

        labels = [[1, 1, 0, 0], [0, 1, 1, 1]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 0.5)

        labels = [[1, 1, 1, 0], [1, 1, 0, 1]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 0)

        labels = [[0, 1, 1, 1], [1, 0, 1, 0]]
        labels = tf.constant(labels, tf.int32)

        with self.test_session():
            self.assertEqual(top_accuracy(outputs, labels).eval(), 1)
