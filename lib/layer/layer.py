import tensorflow as tf

_LAYER_UIDS = {}


def _layer_uid(name):
    if name not in _LAYER_UIDS:
        _LAYER_UIDS[name] = 0

    _LAYER_UIDS[name] += 1
    return _LAYER_UIDS[name]


class Layer(object):
    def __init__(self, placeholders={}, name=None, logging=False):

        if not name:
            layer = self.__class__.__name__.lower()
            name = '{}_{}'.format(layer, _layer_uid(layer))

        self.placeholders = placeholders
        self.name = name
        self.logging = logging
        self.vars = {}

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram('{}/inputs'.format(inputs), inputs)

            outputs = self._call(inputs)

            if self.logging:
                tf.summary.histogram('{}/outputs'.format(outputs), outputs)

            return outputs

    def _call(self, inputs):
        raise NotImplementedError

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram('{}/vars/{}'.format(self.name, var),
                                 self.vars[var])
