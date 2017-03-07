import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from lib.graph.adjacency import grid_adj, normalize_adj, invert_adj
from lib.graph.preprocess import preprocess_adj
from lib.model.layer.fc import FC


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_string('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

mnist = input_data.read_data_sets('data/mnist/', one_hot=False)

WIDTH = 28
HEIGHT = 28
N = HEIGHT * WIDTH
NUM_LABELS = 10
BATCH_SIZE = 128

train_data = mnist.train.images.astype(np.float32)
train_labels = mnist.train.labels

adj = grid_adj([HEIGHT, WIDTH], connectivity=8)
adj = normalize_adj(adj)
adj = invert_adj(adj)
adj = preprocess_adj(adj)  # D^(-1/2) * A * D^(-1/2)

# Build model
x = tf.placeholder(tf.float32, [None, N])
# A = tf.sparse_placeholder(tf.float32, shape=[N, N])
y = tf.placeholder(tf.float32, [None, NUM_LABELS])
dropout = tf.placeholder(tf.float32)

# gcnn1 = GCNN(1, 4, logging=True)
# outputs = gcnn1(x, adj=A)

fc1 = FC(N, 10, dropout=dropout)
outputs = fc1(x)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

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
