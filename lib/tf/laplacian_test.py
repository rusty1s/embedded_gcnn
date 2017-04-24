import numpy as np
import tensorflow as tf


class LaplacianTest(tf.test.TestCase):
    def test_laplacian(self):
        adj = [[0, 1, 0], [1, 0, 2], [0, 2, 0]]
        adj = np.array(adj)

        deg = [1, 3, 2]
        deg = np.array(deg)

        deg_mat = [[1, 0, 0], [0, 3, 0], [0, 0, 2]]
        deg_mat = np.array(deg_mat)

        print(np.dot(adj, deg_mat))
        print(np.dot(adj, deg))
