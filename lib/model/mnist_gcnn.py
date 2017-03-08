from lib.graph.preprocess import preprocess_adj
from lib.graph.adjacency import grid_adj, normalize_adj, invert_adj

adj = grid_adj([HEIGHT, WIDTH], connectivity=8)
adj = normalize_adj(adj)
adj = invert_adj(adj)
adj = preprocess_adj(adj)  # D^(-1/2) * A * D^(-1/2)


