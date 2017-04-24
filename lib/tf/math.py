import numpy as np
import tensorflow as tf


def sparse_identity(size, dtype=tf.float32):
    ones = tf.ones([size], dtype)
    idx = tf.range(0, size, dtype=tf.int64)
    idx = tf.reshape(idx, [-1, 1])
    indices = tf.concat([idx, idx], axis=1)
    return tf.SparseTensorValue(indices, ones, [size, size])


def sparse_scalar_multiply(value, scalar):
    return tf.SparseTensorValue(value.indices, scalar * value.values,
                                value.dense_shape)


def sparse_subtract(a, b):
    c = tf.sparse_add(a, sparse_scalar_multiply(b, -1))
    return tf.SparseTensorValue(c.indices, c.values, a.dense_shape)


def sparse_tensor_diag_matmul(a, diag, transpose=False):
    def _py_func(indices, values, diag):
        diag = diag[indices[:, 1:2]] if transpose else diag[indices[:, 0:1]]
        diag = np.reshape(diag, (-1))
        return np.multiply(values, diag).astype(diag.dtype)

    values = tf.py_func(
        _py_func, [a.indices, a.values, diag], Tout=diag.dtype, stateful=False)

    return tf.SparseTensorValue(a.indices, values, a.dense_shape)
