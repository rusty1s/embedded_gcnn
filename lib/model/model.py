import tensorflow as tf


class Model(object):
    def __init__(self, name=None, logging=False):
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.logging = logging

        self.inputs = None
        self.outputs = None

        self.layers = []
        self.vars = {}
        self.placeholders = {}
        self.optimizer = None
        self.train_op = None
        self.loss = 0

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access.
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Call each layer with the previous outputs.
        self.outputs = self.inputs
        for layer in self.layers:
            self.outputs = layer(self.outputs, self.placeholders)

        # Build metrics.
        self._loss()
        self._accuracy()

        self.train_op = self.optimizer.minimize(self.loss)

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

    def _build(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError
