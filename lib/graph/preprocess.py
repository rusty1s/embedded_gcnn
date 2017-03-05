import numpy as np
import scipy.sparse as sp


def preprocess_adj(adj):
    adj = adj + sp.eye(adj.shape[0])

    deg = np.array(adj.sum(1)).flatten()

    with np.errstate(divide='ignore'):
        deg = np.power(deg, -0.5)
        deg[np.isinf(deg)] = 0

    deg = sp.diags(deg)
    return deg.dot(adj).dot(deg)
