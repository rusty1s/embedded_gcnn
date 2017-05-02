from unittest import TestCase

from numpy.testing import assert_equal

from .grid import grid_points, grid_mass


class GridTest(TestCase):
    def test_grid_points(self):
        points = grid_points((3, 2))
        expected = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
        assert_equal(points, expected)

    def test_grid_mass(self):
        mass = grid_mass((3, 2))
        expected = [1, 1, 1, 1, 1, 1]
        assert_equal(mass, expected)


#     def test_grid_adj_connectivity_8(self):
#         adj = grid_adj((3, 2), connectivity=8)

#       expected = [[0, 1, 1, 2, 0, 0], [1, 0, 2, 1, 0, 0], [1, 2, 0, 1, 1, 2],
#                   [2, 1, 1, 0, 2, 1], [0, 0, 1, 2, 0, 1], [0, 0, 2, 1, 1, 0]]

#         assert_equal(adj.toarray(), expected)
# def test_points_to_embedded_grid_4(self):
#     points = np.array([[2, 2], [2, 3], [3, 2], [2, 1], [1, 2]])
#     adj = sp.coo_matrix([[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
#                          [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
#     adj_dist, adj_rad = points_to_embedded_adj(points, adj)

#     expected_dist = [[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
#                      [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
#     expected_rad = [[0, 2 * np.pi, 0.5 * np.pi, np.pi, 1.5 * np.pi],
#                     [np.pi, 0, 0, 0, 0], [1.5 * np.pi, 0, 0, 0, 0],
#                     [2 * np.pi, 0, 0, 0, 0], [0.5 * np.pi, 0, 0, 0, 0]]

#     assert_equal(adj_dist.toarray(), expected_dist)
#     assert_almost_equal(adj_rad.toarray(), expected_rad, decimal=6)

# def test_points_to_embedded_grid_8(self):
#     points = np.array([[2, 2], [2, 3], [3, 3], [3, 2], [3, 1], [2, 1],
#                        [1, 1], [1, 2], [1, 3]])
#     adj = sp.coo_matrix(
#         [[0, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
#          [1, 0, 0, 0, 0, 0, 0, 0, 0]])
#     adj_dist, adj_rad = points_to_embedded_adj(points, adj)

#     expected_dist = [
#         [0, 1, 2, 1, 2, 1, 2, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [2, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [2, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [2, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [2, 0, 0, 0, 0, 0, 0, 0, 0]
#     ]
#     expected_rad = [[
#         0, 2 * np.pi, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi,
#         1.25 * np.pi, 1.5 * np.pi, 1.75 * np.pi
#     ], [np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [1.25 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [1.5 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [1.75 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [2 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0.25 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0.5 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [0.75 * np.pi, 0, 0, 0, 0, 0, 0, 0, 0]]

#     assert_equal(adj_dist.toarray(), expected_dist)
#     assert_almost_equal(adj_rad.toarray(), expected_rad, decimal=6)
