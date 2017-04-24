import tensorflow as tf


def sparse_identity(size, dtype=tf.float32):
    ones = tf.ones([size], dtype=dtype)
    idx = tf.range(0, size, dtype=tf.int64)
    idx = tf.reshape(idx, [-1, 1])
    indices = tf.concat([idx, idx], axis=1)
    return tf.SparseTensor(indices, ones, [size, size])


def sparse_scalar_multiply(value, scalar):
    return tf.SparseTensor(value.indices, scalar * value.values,
                           value.dense_shape)


def sparse_subtract(a, b):
    return tf.sparse_add(a, sparse_scalar_multiply(b, -1))


def sparse_tensor_diag_matmul(a, diag, transpose=False):
    raise NotImplementedError
