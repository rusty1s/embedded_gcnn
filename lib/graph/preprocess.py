import numpy as np
import scipy.sparse as sp


def preprocess_adj(adj):
    adj = adj + sp.eye(adj.shape[0], dtype=adj.dtype)
    degree = np.array(adj.sum(1)).flatten()
    degree = np.power(degree, -0.5)
    degree = sp.diags(degree)
    return degree.dot(adj).dot(degree)
