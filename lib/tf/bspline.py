from __future__ import division

import tensorflow as tf
from numpy import pi as PI


def base(adj_rad, K, P, p):
    assert p < P

    shape = adj_rad.dense_shape
    dtype = tf.as_dtype(adj_rad.values.dtype)

    c = P / (2 * PI)  # Don't recalculate coefficient every time.
    values = adj_rad.values
    indices = adj_rad.indices

    if K == 1:
        values = c * values - p

        greater_zero = tf.greater(values, tf.zeros_like(values))
        less_equal_one = tf.less_equal(values, tf.ones_like(values))

        values = tf.logical_and(greater_zero, less_equal_one)
        values = tf.cast(values, dtype)

        return tf.SparseTensorValue(indices, values, shape)

    elif K == 2:
        zero = tf.SparseTensorValue(
            tf.constant(0, dtype=tf.int64, shape=[0, 2]),
            tf.constant(0, dtype, shape=[0]), shape)

        left_values = c * values - p
        left = tf.SparseTensorValue(indices, left_values, shape)
        left = tf.sparse_maximum(left, zero)

        right_values = -c * values + p + 2
        right = tf.SparseTensorValue(indices, right_values, shape)
        right = tf.sparse_maximum(right, zero)

        offset_right_values = right_values - P
        offset_right = tf.SparseTensorValue(indices, offset_right_values,
                                            shape)
        offset_right = tf.sparse_maximum(offset_right, zero)

        return tf.sparse_maximum(tf.sparse_minimum(left, right), offset_right)

    else:
        raise NotImplementedError
