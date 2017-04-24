from .math import sparse_identity, sparse_subtract
from .adjacency import normalize_adj


def laplacian(adj):
    adj_norm = normalize_adj(adj)

    N = adj.get_shape()[0].value
    I = sparse_identity(N, adj.dtype)

    return sparse_subtract(I, adj_norm)


def rescale_lap(lap):
    N = lap.get_shape()[0].value
    I = sparse_identity(N, lap.dtype)
    return sparse_subtract(lap, I)
