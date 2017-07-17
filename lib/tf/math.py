import numpy as np
import tensorflow as tf


def sparse_scalar_multiply(value, scalar):
    """Multiply a SparseTensorValue by a scalar."""

    return tf.SparseTensorValue(value.indices, scalar * value.values,
                                value.dense_shape)


def sparse_tensor_diag_matmul(a, diag, transpose=False):
    """Multiply a SparseTensorValue with a diagonal matrix given its diagonal
    values."""

    _py_func = _diag_matmul_py if not transpose else _diag_matmul_transpose_py
    values = tf.py_func(
        _py_func, [a.indices, a.values, diag], Tout=diag.dtype, stateful=False)

    return tf.SparseTensorValue(a.indices, values, a.dense_shape)


def _diag_matmul_py(indices, values, diag):
    diag = diag[indices[:, 0:1]]
    diag = np.reshape(diag, (-1))
    return np.multiply(values, diag).astype(diag.dtype)


def _diag_matmul_transpose_py(indices, values, diag):
    diag = diag[indices[:, 1:2]]
    diag = np.reshape(diag, (-1))
    return np.multiply(values, diag).astype(diag.dtype)
