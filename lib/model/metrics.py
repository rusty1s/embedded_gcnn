import tensorflow as tf


def softmax_cross_entropy(outputs, labels):
    """Calculate softmax cross-entropy loss."""

    loss_per_example = tf.nn.softmax_cross_entropy_with_logits(
        logits=outputs, labels=labels, name='loss_per_example')
    return tf.reduce_mean(loss_per_example, name='loss')


def sigmoid_cross_entropy(outputs, labels):
    """Calculate sigmoid cross-entropy loss."""

    labels = tf.cast(labels, tf.float32)
    loss_per_example = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=outputs, labels=labels, name='loss_per_example')
    return tf.reduce_mean(loss_per_example, name='loss')


def total_loss(loss):
    tf.add_to_collection('losses', loss)
    losses = tf.get_collection('losses')

    for loss in losses:
        tf.summary.scalar(loss.op.name, loss)

    return tf.add_n(losses, name='total_loss')


def top_accuracy(outputs, labels):
    """Calculate accuracy."""

    num_labels = labels.get_shape()[1]

    with tf.name_scope('accuracy'):
        labels = tf.cast(labels, tf.bool)

        predicted_labels = tf.argmax(outputs, axis=1)
        predicted_labels_one_hot = tf.one_hot(predicted_labels, num_labels)
        predicted_labels_one_hot = tf.cast(predicted_labels_one_hot, tf.bool)

        correct_prediction = tf.logical_and(predicted_labels_one_hot, labels)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        correct_prediction = tf.reduce_max(correct_prediction, axis=1)

        accuracy = tf.reduce_mean(correct_prediction)

    tf.summary.scalar('train_accuracy', accuracy)

    return accuracy
