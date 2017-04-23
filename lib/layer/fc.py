import tensorflow as tf

from .var_layer import VarLayer


class FC(VarLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0.0,
                 **kwargs):

        super(FC, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.dropout = dropout

    def _call(self, inputs):
        outputs = tf.reshape(inputs, [-1, self.in_channels])

        outputs = tf.nn.dropout(outputs, 1 - self.dropout)
        outputs = tf.matmul(outputs, self.vars['weights'])

        if self.bias:
            outputs = tf.nn.bias_add(outputs, self.vars['bias'])

        return self.act(outputs)
