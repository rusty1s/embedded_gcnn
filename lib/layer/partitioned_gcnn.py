from six.moves import xrange

import tensorflow as tf

from .layer import Layer
from .inits import weight_variable, bias_variable


class PartitionedGCNN(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 adjs,
                 num_partitions,
                 weight_stddev=0.1,
                 weight_decay=None,
                 bias=True,
                 bias_constant=0.1,
                 act=tf.nn.relu,
                 **kwargs):

        super(PartitionedGCNN, self).__init__(**kwargs)

        self.adjs = adjs
        self.bias = bias
        self.act = act

        with tf.variable_scope('{}_vars'.format(self.name)):
            self.vars['weights'] = weight_variable(
                [num_partitions, in_channels, out_channels],
                '{}_weights'.format(self.name), weight_stddev, weight_decay)

            if self.bias:
                self.vars['bias'] = bias_variable(
                    [out_channels], '{}_bias'.format(self.name), bias_constant)

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # multiple = isinstance(inputs, list)
        num_partitions = self.vars['weights'].get_shape()[0].value
        # in_channels = self.vars['weights'].get_shape()[1].value
        # out_channels = self.vars['weights'].get_shape()[2].value
        batch_size = len(inputs)

        outputs = list()

        for i in xrange(batch_size):
            output = 0
            for j in xrange(num_partitions):
                x = tf.sparse_tensor_dense_matmul(self.adjs[i][j], inputs[i])
                x = tf.matmul(x, self.vars['weights'][j])
                output += x

            if self.bias:
                output = tf.nn.bias_add(output, self.vars['bias'])

            outputs.append(output)

        return outputs

#         n = inputs.get_shape()[1].value
#         in_channels = inputs.get_shape()[2].value
#         out_channels = self.vars['weights'].get_shape()[2].value
#         multiple = isinstance(self.adjs[0], (list, tuple))

#         outputs = list()
#         for i in xrange(inputs.get_shape()[0].value):
#             adjs = self.adjs[i] if multiple else self.adjs

#             output = tf.zeros([n, out_channels], dtype=inputs.dtype)
#             for j in xrange(self.vars['weights'].get_shape()[0].value):
#                 x = tf.zeros([n, in_channels], dtype=inputs.dtype)
#                 x += tf.sparse_tensor_dense_matmul(adjs[j], inputs[i])
#                 x = tf.matmul(x, self.vars['weights'][j])
#                 output += x

#             outputs.append(output)

#         outputs = tf.stack(outputs, axis=0)

#         if self.bias:
#             outputs = tf.nn.bias_add(outputs, self.vars['bias'])

#         return self.act(outputs)
