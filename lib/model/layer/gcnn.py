import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class GCNN(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_stddev=0.0,
                 weight_decay=None,
                 bias=False,
                 bias_constant=0.0,
                 act=tf.nn.relu,
                 **kwargs):

        super().__init__(**kwargs)

        self.act = act
        self.bias = bias

        with tf.variable_scope('{}_vars'.format(self.name)):
            self.vars['weights'] = weight_variable(
                [in_channels, out_channels],
                weight_stddev,
                weight_decay,
                name='{}_weights'.format(self.name))

            if self.bias:
                self.vars['bias'] = bias_variable(
                    [out_channels],
                    bias_constant,
                    name='{}_bias'.format(self.name))

        if self.logging:
            self._log_vars()

    def _call(self, inputs, **kwargs):
        X = tf.sparse_tensor_dense_matmul(kwargs.get('A'), inputs)
        outputs = tf.matmul(X, self.vars['weights'])

        if self.bias:
            outputs += self.vars['bias']

        return self.act(outputs)
