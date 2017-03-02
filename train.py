import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from embedded_gcnn import laplacian


mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

train_data = mnist.train.images.astype(np.float32)
print(mnist)
print(train_data.shape)



