import time

import tensorflow as tf

from .metrics import cal_softmax_cross_entropy, cal_accuracy


class Model(object):
    def __init__(self,
                 placeholders,
                 name=None,
                 learning_rate=0.001,
                 train_dir=None,
                 log_dir=None):

        if not name:
            name = self.__class__.__name__.lower()

        self.placeholders = placeholders
        self.name = name
        self.train_dir = train_dir
        self.log_dir = log_dir
        self.logging = False if log_dir is None else True
        self.sess = None

        self.inputs = placeholders['features']
        self.labels = placeholders['labels']
        self.outputs = None

        self.layers = []
        self.vars = {}

        self.optimizer = tf.train.AdamOptimizer(learning_rate)

        self._accuracy = 0
        self._loss = 0
        self._train = None
        self._summary = None
        self._writer = None

        # Create global step variable.
        self._global_step = tf.get_variable(
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
        self._loss = cal_softmax_cross_entropy(self.outputs, self.labels)
        self._accuracy = cal_accuracy(self.outputs, self.labels)

        # Build train op.
        self._train = self.optimizer.minimize(
            self._loss, global_step=self._global_step)

        # Create session.
        self.sess = tf.Session()
        if self.log_dir is not None:
            self._summary = tf.summary.merge_all()
            self._writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def _build(self):
        raise NotImplementedError

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

        if self.train_dir is None:
            return self.sess.run(self._global_step)

        if tf.gfile.Exists(self.train_dir):
            saver = tf.train.Saver(self.vars)
            save_path = '{}/checkpoint.ckpt'.format(self.train_dir)
            saver.restore(self.sess, save_path)
            print('Model restored from file {}.'.format(save_path))
        else:
            tf.gfile.MakeDirs(self.train_dir)

        return self.sess.run(self.global_step)

    def save(self):
        if self.train_dir is None:
            return

        saver = tf.train.Saver(self.vars)
        save_path = '{}/checkpoint.ckpt'.format(self.train_dir)
        saver.save(self.sess, save_path)
        print('Model saved in file {}.'.format(save_path))

    def train(self, feed_dict, step=None):
        t = time.time()

        if self.log_dir is None:
            self.sess.run(self._train, feed_dict)
        else:
            _, summary = self.sess.run([self._train, self._summary], feed_dict)
            self._writer.add_summary(summary, step)

        return time.time() - t

    def evaluate(self, feed_dict):
        t = time.time()
        loss, acc = self.sess.run([self._loss, self._accuracy], feed_dict)
        return loss, acc, t - time.time()
