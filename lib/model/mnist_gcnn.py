from __future__ import division

from .model import Model
from ..layer.gcnn import GCNN as Conv
from ..layer.max_pool_gcnn import MaxPoolGCNN as MaxPool
from ..layer.fc import FC


class MNIST_GCNN(Model):
    def __init__(self, **kwargs):
        # 5 placeholders:
        #################
        # * features
        # * labels
        # * adjacency_1
        # * adjacency_2
        # * dropout
        super(MNIST_GCNN, self).__init__(**kwargs)
        self.build()

    def _build(self):
        n = self.placeholders['adjacency_1'].get_shape()[0].value

        conv_1 = Conv(
            1, 32, self.placeholders['adjacency_1'], logging=self.logging)
        max_pool_1 = MaxPool(size=4, logging=self.logging)
        conv_2 = Conv(
            32, 64, self.placeholders['adjacency_2'], logging=self.logging)
        max_pool_2 = MaxPool(size=4, logging=self.logging)
        fc_1 = FC(n // 4 // 4 * 64, 1024, logging=self.logging)
        fc_2 = FC(1024,
                  10,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [conv_1, max_pool_1, conv_2, max_pool_2, fc_1, fc_2]
