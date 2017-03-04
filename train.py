# import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from embedded_gcnn import laplacian, chebyshev
from embedded_gcnn import grid

mnist = input_data.read_data_sets('data/mnist/', one_hot=False)

train_data = mnist.train.images.astype(np.float32)
train_labels = mnist.train.labels

WIDTH = 28
HEIGHT = 28
K = 1  # only local neighbors

A = grid((HEIGHT, WIDTH), connectivity=8)
L = laplacian(A, normalized=True)
X = train_data[0]
T_k = chebyshev(L, X, K, normalized=True)
