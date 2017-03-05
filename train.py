import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/mnist/', one_hot=False)

train_data = mnist.train.images.astype(np.float32)
train_labels = mnist.train.labels

WIDTH = 28
HEIGHT = 28
k = 1  # only local neighbors

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_string('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for epoch in xrange(FLAGS.epochs):
#     t = time.time()
#     feed_dict.update({placeholders['dropout']: FLAGS.dropout})

#     # Training step
# outs = sess.run([model.opt, model.loss, model.accurary],
#                 feed_dict=feed_dict)

#     # Validation

# print('Optimization finished!')

# Testing
# cost, acc, duration = evaluate()
# print('Test set results: cost={:.5f}, accuracy={:.5f}, time={:.5f}'.format(
#     cost, acc, duration))
