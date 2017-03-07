import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from lib.graph.adjacency import grid_adj, normalize_adj, invert_adj
from lib.graph.preprocess import preprocess_adj
from lib.layer.fc import FC
from lib.model.model import Model

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

# train_data = mnist.train.images.astype(np.float32)
# train_labels = mnist.train.labels

adj = grid_adj([HEIGHT, WIDTH], connectivity=8)
adj = normalize_adj(adj)
adj = invert_adj(adj)
adj = preprocess_adj(adj)  # D^(-1/2) * A * D^(-1/2)

placeholders = {
    'features': tf.placeholder(tf.float32, shape=[None, N]),
    'labels': tf.placeholder(tf.int32, shape=[None]),
    'dropout': tf.placeholder_with_default(0.0, shape=[]),
}


class MNIST(Model):
    def __init__(self, **kwargs):
        super(MNIST, self).__init__(**kwargs)
        self.build()

    def _build(self):
        self.layers.append(FC(N, NUM_LABELS))


model = MNIST(placeholders=placeholders, logging=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20000):
    xs, ys = mnist.train.next_batch(2000)
    feed_dict = {
        placeholders['features']: xs,
        placeholders['labels']: ys,
        placeholders['dropout']: 0.5,
    }
    _, loss, acc = sess.run([model.train, model.loss, model.accuracy],
                            feed_dict)

    # Print results.
    print(', '.join([
        'Step: {}'.format(step), 'train_loss={:.5f}'.format(loss),
        'train_acc={:.5f}'.format(acc)
    ]))

print('Optimization finished!')

# Testing.
# cost, acc, duration = evaluate()
# print('Test set results: cost={:.5f}, accuracy={:.5f}, time={:.5f}'.format(
#     cost, acc, duration))
