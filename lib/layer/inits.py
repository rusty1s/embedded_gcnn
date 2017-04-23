import tensorflow as tf


def weight_variable(shape,
                    name,
                    stddev=0.1,
                    decay=0.0,
                    dtype=tf.float32):
    initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    var = tf.get_variable(name, shape, dtype, initializer)

    if decay > 0:
        decay = tf.multiply(tf.nn.l2_loss(var), decay, name='weight_loss')
        tf.add_to_collection('losses', decay)

    return var


def bias_variable(shape, name, constant=0.1, dtype=tf.float32):
    initializer = tf.constant_initializer(constant, dtype)
    return tf.get_variable(name, shape, dtype, initializer)
