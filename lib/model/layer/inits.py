import tensorflow as tf


def weight_variable(shape,
                    stddev=0.01,
                    decay=None,
                    name=None,
                    dtype=tf.float32):
    initial = tf.truncated_normal_initializer(stddev, dtype=dtype)
    var = tf.get_variable(name, shape, dtype, initializer=initial)

    if decay is not None:
        decay = tf.multiply(tf.nn.l2_loss(var), decay, name='weight_loss')
        tf.add_to_collection('losses', decay)

    return var


def bias_variable(shape, constant=0.0, name=None, dtype=tf.float32):
    initial = tf.constant_initializer(constant)
    return tf.get_variable(name, shape, dtype, initializer=initial)
