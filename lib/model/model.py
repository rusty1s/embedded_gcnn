import tensorflow as tf

from .metrics import cal_softmax_cross_entropy, cal_accuracy


class Model(object):
    def __init__(self,
                 placeholders,
                 name=None,
                 learning_rate=0.001,
                 logging=False):

        if not name:
            name = self.__class__.__name__.lower()

        self.placeholders = placeholders
        self.name = name
        self.logging = logging

        self.inputs = placeholders['features']
        self.labels = placeholders['labels']
        self.outputs = None

        self.layers = []
        self.vars = {}

        self.optimizer = tf.train.AdamOptimizer(learning_rate)

        self.accuracy = 0
        self.loss = 0
        self.train = None

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for saving and loading.
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Call each layer with the previous outputs.
        print(self.placeholders['features'])
        self.outputs = self.inputs
        for layer in self.layers:
            self.outputs = layer(self.outputs)

        # Build metrics.
        self.loss = cal_softmax_cross_entropy(self.outputs, self.labels)
        self.accuracy = cal_accuracy(self.outputs, self.labels)

        self.train = self.optimizer.minimize(self.loss)

    def _build(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError('TensorFlow session not provided.')

        saver = tf.train.Saver(self.vars)
        save_path = 'data/{}/checkpoint.ckpt'.format(self.name)
        saver.save(sess, save_path)
        print('Model saved in file {}.'.format(save_path))

    def load(self, sess=None):
        if not sess:
            raise AttributeError('TensorFlow session not provided.')

        saver = tf.train.Saver(self.vars)
        save_path = 'data/{}/checkpoint.ckpt'.format(self.name)
        saver.restore(sess, save_path)
        print('Model restored from file {}.'.format(save_path))
