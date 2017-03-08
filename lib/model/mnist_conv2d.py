import tensorflow as tf

from .model import Model
from ..layer.conv2d import Conv2d
from ..layer.max_pool2d import MaxPool2d
from ..layer.fc import FC


class MNISTConv2d(Model):
    def __init__(self, **kwargs):
        # 3 placeholders: features, labels, dropout
        super(MNISTConv2d, self).__init__(**kwargs)
        self.build()

    def _preprocess(self):
        return tf.reshape(self.inputs, [-1, 28, 28, 1])

    def _build(self):
        conv1 = Conv2d(1, 32, size=5, stride=1, logging=self.logging)
        pool1 = MaxPool2d(size=2, stride=2, logging=self.logging)
        conv2 = Conv2d(32, 64, size=5, stride=1, logging=self.logging)
        pool2 = MaxPool2d(size=2, stride=2, logging=self.logging)
        fc1 = FC(7 * 7 * 64, 1024, logging=self.logging)
        fc2 = FC(1024,
                 10,
                 dropout=self.placeholders['dropout'],
                 act=lambda x: x,
                 logging=self.logging)

        self.layers = [conv1, pool1, conv2, pool2, fc1, fc2]
