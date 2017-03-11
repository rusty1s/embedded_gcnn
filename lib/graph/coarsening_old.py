from six.moves import xrange

import numpy as np
import scipy.sparse
import scipy.sparse as sp

from .distortion import perm_adj


def coarsen_adj(adj, levels):
    assert levels > 0, 'Levels must be greather than zero.'

    # Generate levels + 1 graphs.
    adjs, parents = graclus(adj, levels)

    # Sort adjacencies so we can perform an efficient pooling operation.
    perms = _compute_perms(parents)
    adjs = [perm_adj(adjs[i], perms[i]) for i in xrange(len(adjs))]

    # Return both adjacencies and permutation to sort node features.
    return adjs, perms[0]




# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def metis_one_level(rr, cc, vv, rid, weights):

    nnz = rr.shape[0]
    N = rr[nnz - 1] + 1

    marked = np.zeros(N, np.bool)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count + 1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs + jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[rs + jj] * (
                        1.0 / weights[tid] + 1.0 / weights[nid])
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid

            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True

            clustercount += 1

    return cluster_id



def graclus(W, levels, rid=None):
    """Coarsen a graph multiple times using the GRACLUS algorithm."""

    if rid is None:
        rid = np.random.permutation(np.arange(W.shape[0]))

    parents = []
    graphs = [W]

    # degree = np.array(W.sum(axis=1)).flatten()

    for _ in xrange(levels):

        # CHOOSE THE WEIGHTS FOR THE PAIRING
        # weights = ones(N,1)       # metis weights
        # weights = degree  # graclus weights

        cluster_id = cluster_adj_once(W, rid)


        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = scipy.sparse.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        # cluster_id = metis_one_level(rr, cc, vv, rid, weights)  # rr is ordered
        # maximal zwei Knoten werden jeweils eine Cluster ID zugeordnet, aufsteigend
        print('cluster', cluster_id)
        parents.append(cluster_id)

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = scipy.sparse.csr_matrix((nvv, (nrr, ncc)), shape=(Nnew, Nnew))
        W.eliminate_zeros()
        # Add new graph to the list of all coarsened graphs
        graphs.append(W)
        N, N = W.shape

        # # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS)
        # degree = np.array(W.sum(axis=0)).flatten()
        #degree = W.sum(axis=0) - W.diagonal()

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        #[~, rid]=sort(ss);     # arthur strategy
        #[~, rid]=sort(supernode_size);    #  thomas strategy
        #rid=randperm(N);                  #  metis/graclus strategy
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents




def _compute_perms(parents):
    """Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree."""

    M_last = max(parents[-1]) + 1
    indices = [np.arange(M_last)]

    for parent in parents[::-1]:
        #print('parent: {}'.format(parent))

        # Fake nodes go after real ones.
        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2
            #print('indices_node: {}'.format(indices_node))

            # Add a node to go with a singelton.
            if len(indices_node) is 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1
                #print('new singelton: {}'.format(indices_node))
            # Add two nodes as children of a singelton in the parent.
            elif len(indices_node) is 0:
                indices_node.append(pool_singeltons + 0)
                indices_node.append(pool_singeltons + 1)
                pool_singeltons += 2
                #print('singelton childrens: {}'.format(indices_node))

            indices_layer.extend(indices_node)
        indices.append(np.array(indices_layer))

    # Sanity checks.
    for i, indices_layer in enumerate(indices):
        M = M_last * 2**i
        # Reduction by 2 at each layer (binary tree).
        assert len(indices[0] == M)
        # The new ordering does not omit an indice.
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1]


# assert (compute_perm([np.array([4,1,1,2,2,3,0,0,3]),np.array([2,1,0,1,0])])
#         == [[3,4,0,9,1,2,5,8,6,7,10,11],[2,4,1,3,0,5],[0,1,2]])
