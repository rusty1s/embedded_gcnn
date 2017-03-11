import tensorflow as tf

from .model import Model
from ..layer.conv2d import Conv2d as Conv
from ..layer.max_pool2d import MaxPool2d as MaxPool
from ..layer.fc import FC


class MNISTConv2d(Model):
    def __init__(self, **kwargs):
        # 3 placeholders: features, labels, dropout
        super(MNISTConv2d, self).__init__(**kwargs)
        self.build()

    def _preprocess(self):
        return tf.reshape(self.inputs, [-1, 28, 28, 1])

    def _build(self):
        conv_1 = Conv(1, 32, size=5, stride=1, logging=self.logging)
        pool_1 = MaxPool(size=2, stride=2, logging=self.logging)
        conv_2 = Conv(32, 64, size=5, stride=1, logging=self.logging)
        pool_2 = MaxPool(size=2, stride=2, logging=self.logging)
        fc_1 = FC(7 * 7 * 64, 1024, logging=self.logging)
        fc_2 = FC(1024,
                  10,
                  dropout=self.placeholders['dropout'],
                  act=lambda x: x,
                  logging=self.logging)

        self.layers = [conv_1, pool_1, conv_2, pool_2, fc_1, fc_2]
