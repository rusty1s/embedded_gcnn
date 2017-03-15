import scipy.sparse as sp
import tensorflow as tf

from .sparse import sparse_to_tensor


class SparseTest(tf.test.TestCase):
    def test_sparse_to_tensor(self):
        value = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        value = sp.coo_matrix(value)

        with self.test_session():
            self.assertAllEqual(
                tf.sparse_tensor_to_dense(sparse_to_tensor(value)).eval(),
                value.toarray())

    def test_sparse_feed_dict(self):
        value = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        value = sp.coo_matrix(value)
        value = sparse_to_tensor(value)

        # Sparse placeholder is buggy and can't convert shape.
        # => Need to pass empty shape.
        placeholder = tf.sparse_placeholder(tf.float32)
        output = tf.sparse_tensor_to_dense(placeholder)

        expected = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]

        with self.test_session() as sess:
            result = sess.run(output, feed_dict={placeholder: value})
            self.assertAllEqual(result, expected)
