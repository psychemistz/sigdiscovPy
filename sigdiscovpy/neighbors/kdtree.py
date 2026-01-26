"""
KD-tree based neighbor search for scalable spatial analysis.

Uses scipy.spatial.cKDTree for O(n log n) neighbor queries.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree


class KDTreeNeighborSearch:
    """
    KD-tree based spatial neighbor search.

    Provides efficient O(n log n) neighbor queries for large datasets.
    Used for streaming computation of spatial weights.

    Parameters
    ----------
    coords : array-like
        Spatial coordinates of shape (n, 2).
    leafsize : int, default=16
        Number of points at which to switch to brute-force search.

    Examples
    --------
    >>> coords = np.random.randn(10000, 2) * 100
    >>> searcher = KDTreeNeighborSearch(coords)
    >>> neighbors = searcher.query_radius(50)
    >>> len(neighbors)  # List of neighbor indices for each point
    10000
    """

    def __init__(self, coords, leafsize: int = 16):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.n_points = self.coords.shape[0]
        self.tree = cKDTree(self.coords, leafsize=leafsize)

    def query_radius(
        self,
        radius: float,
        return_distance: bool = False,
    ) -> List:
        """
        Find all neighbors within radius for each point.

        Parameters
        ----------
        radius : float
            Search radius.
        return_distance : bool, default=False
            Whether to also return distances.

        Returns
        -------
        list
            List of arrays, where neighbors[i] contains indices of
            points within radius of point i.
            If return_distance=True, returns (neighbors, distances).
        """
        if return_distance:
            return self.tree.query_ball_point(
                self.coords,
                radius,
                return_sorted=True,
            )
        else:
            return self.tree.query_ball_tree(self.tree, radius)

    def query_radius_sparse(
        self,
        radius: float,
        exclude_self: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find neighbors and return in sparse coordinate format.

        Parameters
        ----------
        radius : float
            Search radius.
        exclude_self : bool, default=True
            Exclude self-connections (diagonal).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (rows, cols, distances) in COO format.
        """
        neighbors = self.tree.query_ball_tree(self.tree, radius)

        rows_list = []
        cols_list = []
        dists_list = []

        for i, neighbor_indices in enumerate(neighbors):
            for j in neighbor_indices:
                if exclude_self and i == j:
                    continue
                d = np.linalg.norm(self.coords[i] - self.coords[j])
                rows_list.append(i)
                cols_list.append(j)
                dists_list.append(d)

        return (
            np.array(rows_list, dtype=np.int32),
            np.array(cols_list, dtype=np.int32),
            np.array(dists_list, dtype=np.float64),
        )

    def query_directional(
        self,
        sender_coords,
        radius: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find receiver neighbors for each sender point.

        Parameters
        ----------
        sender_coords : array-like
            Coordinates of sender points (n_senders, 2).
        radius : float
            Search radius.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (sender_indices, receiver_indices, distances)
        """
        sender_coords = np.asarray(sender_coords, dtype=np.float64)
        neighbors = self.tree.query_ball_point(sender_coords, radius)

        rows_list = []
        cols_list = []
        dists_list = []

        for i, neighbor_indices in enumerate(neighbors):
            for j in neighbor_indices:
                d = np.linalg.norm(sender_coords[i] - self.coords[j])
                rows_list.append(i)
                cols_list.append(j)
                dists_list.append(d)

        return (
            np.array(rows_list, dtype=np.int32),
            np.array(cols_list, dtype=np.int32),
            np.array(dists_list, dtype=np.float64),
        )


def query_radius_neighbors(
    coords,
    radius: float,
    exclude_self: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function for radius neighbor query.

    Parameters
    ----------
    coords : array-like
        Spatial coordinates (n, 2).
    radius : float
        Search radius.
    exclude_self : bool, default=True
        Exclude self-connections.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (rows, cols, distances) in COO format.

    Examples
    --------
    >>> coords = np.random.randn(1000, 2) * 100
    >>> rows, cols, dists = query_radius_neighbors(coords, radius=50)
    """
    searcher = KDTreeNeighborSearch(coords)
    return searcher.query_radius_sparse(radius, exclude_self=exclude_self)
