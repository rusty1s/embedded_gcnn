import tensorflow as tf


def cal_softmax_cross_entropy(outputs, labels):
    """Calculate softmax cross-entropy loss."""

    loss_per_example = tf.nn.softmax_cross_entropy_with_logits(
        logits=outputs, labels=labels, name='loss_per_example')
    loss = tf.reduce_mean(loss_per_example, name='loss')
    tf.add_to_collection('losses', loss)

    losses = tf.get_collection('losses')

    for loss in losses:
        tf.summary.scalar(loss.op.name, loss)

    return tf.add_n(losses, name='total_loss')


def cal_accuracy(outputs, labels):
    """Calculate accuracy."""

    with tf.name_scope('accuracy'):
        predicted_labels = tf.argmax(outputs, 1)
        predicted_labels = tf.cast(predicted_labels, tf.int32)

        # TODO: Multi label problem
        correct_labels = tf.argmax(labels, 1)
        correct_labels = tf.cast(correct_labels, tf.int32)

        correct_prediction = tf.equal(predicted_labels, correct_labels)
        correct_prediction = tf.cast(correct_prediction, tf.float32)

        accuracy = tf.reduce_mean(correct_prediction)

    tf.summary.scalar('train_accuracy', accuracy)
    return accuracy
