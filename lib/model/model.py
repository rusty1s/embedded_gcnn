import tensorflow as tf

from .metrics import cal_softmax_cross_entropy, cal_accuracy


class Model(object):
    def __init__(self,
                 placeholders,
                 name=None,
                 learning_rate=0.001,
                 train_dir=None,
                 logging=False):

        if not name:
            name = self.__class__.__name__.lower()

        self.placeholders = placeholders
        self.name = name
        self.train_dir = train_dir
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
        self.summary = None

        # Create global step variable.
        self.global_step = tf.get_variable(
            '{}/global_step'.format(self.name),
            shape=[],
            dtype=tf.int32,
            initializer=tf.constant_initializer(0, dtype=tf.int32),
            trainable=False)

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for saving and loading.
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Call each layer with the previous outputs.
        self.outputs = self.inputs
        for layer in self.layers:
            self.outputs = layer(self.outputs)

        # Build metrics.
        self.loss = cal_softmax_cross_entropy(self.outputs, self.labels)
        self.accuracy = cal_accuracy(self.outputs, self.labels)

        self.train = self.optimizer.minimize(
            self.loss,
            global_step=self.global_step)
        self.summary = tf.summary.merge_all()

    def _build(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError('TensorFlow session not provided.')

        if self.train_dir is None:
            return

        saver = tf.train.Saver(self.vars)
        save_path = '{}/checkpoint.ckpt'.format(self.train_dir)
        saver.save(
            sess,
            save_path)
        print('Model saved in file {}.'.format(save_path))

    def initialize(self, sess=None):
        if not sess:
            raise AttributeError('TensorFlow session not provided.')

        sess.run(tf.global_variables_initializer())

        if self.train_dir is None:
            return sess.run(self.global_step)

        if tf.gfile.Exists(self.train_dir):
            saver = tf.train.Saver(self.vars)
            save_path = '{}/checkpoint.ckpt'.format(self.train_dir)
            saver.restore(sess, save_path)
            print('Model restored from file {}.'.format(save_path))
        else:
            tf.gfile.MakeDirs(self.train_dir)

        return sess.run(self.global_step)
