"""
Spatial layout generation for simulation from real data patterns.

Extracts spatial patterns from real data to inform simulations.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import distance

from .loaders import SpatialData


@dataclass
class SpatialLayout:
    """Spatial layout extracted from real data."""

    coords: np.ndarray  # (n_cells, 2)
    cell_types: np.ndarray  # (n_cells,)
    sender_indices: np.ndarray  # indices of sender cells
    receiver_indices: np.ndarray  # indices of receiver cells
    domain_radius: float
    center: tuple[float, float]
    density: float  # cells per unit area

    @property
    def n_senders(self) -> int:
        return len(self.sender_indices)

    @property
    def n_receivers(self) -> int:
        return len(self.receiver_indices)


class SpatialLayoutGenerator:
    """
    Generate spatial layouts from real data patterns.

    Extracts cell type distributions and spatial arrangements
    that can be used to inform simulations.
    """

    def __init__(self, data: SpatialData):
        """
        Initialize with real spatial data.

        Parameters
        ----------
        data : SpatialData
            Real spatial transcriptomics data.
        """
        self.data = data
        self._compute_domain_stats()

    def _compute_domain_stats(self):
        """Compute domain statistics from data."""
        self.center = np.mean(self.data.coords, axis=0)
        distances = np.linalg.norm(self.data.coords - self.center, axis=1)
        self.max_radius = np.max(distances)
        self.domain_area = np.pi * self.max_radius**2
        self.density = self.data.n_cells / self.domain_area

    def extract_layout(
        self,
        sender_type: str,
        receiver_type: str,
    ) -> SpatialLayout:
        """
        Extract spatial layout for sender-receiver analysis.

        Parameters
        ----------
        sender_type : str
            Cell type label for senders.
        receiver_type : str
            Cell type label for receivers.

        Returns
        -------
        SpatialLayout
            Extracted spatial layout.
        """
        if self.data.cell_types is None:
            raise ValueError("Data does not have cell type annotations")

        sender_mask = self.data.cell_types == sender_type
        receiver_mask = self.data.cell_types == receiver_type

        sender_indices = np.where(sender_mask)[0]
        receiver_indices = np.where(receiver_mask)[0]

        return SpatialLayout(
            coords=self.data.coords.copy(),
            cell_types=self.data.cell_types.copy(),
            sender_indices=sender_indices,
            receiver_indices=receiver_indices,
            domain_radius=self.max_radius,
            center=tuple(self.center),
            density=self.density,
        )

    def compute_celltype_fractions(self) -> dict[str, float]:
        """
        Compute fraction of each cell type.

        Returns
        -------
        dict
            Cell type -> fraction mapping.
        """
        if self.data.cell_types is None:
            raise ValueError("Data does not have cell type annotations")

        unique, counts = np.unique(self.data.cell_types, return_counts=True)
        total = self.data.n_cells

        return {ct: count / total for ct, count in zip(unique, counts)}

    def compute_neighbor_distances(
        self,
        cell_type: Optional[str] = None,
        k: int = 10,
    ) -> np.ndarray:
        """
        Compute k-nearest neighbor distances.

        Parameters
        ----------
        cell_type : str, optional
            Restrict to this cell type. If None, use all cells.
        k : int
            Number of nearest neighbors.

        Returns
        -------
        np.ndarray
            Distance to k-th nearest neighbor for each cell.
        """
        if cell_type is not None and self.data.cell_types is not None:
            mask = self.data.cell_types == cell_type
            coords = self.data.coords[mask]
        else:
            coords = self.data.coords

        # Compute pairwise distances
        dists = distance.cdist(coords, coords)

        # Get k-th nearest (excluding self)
        k_distances = np.zeros(len(coords))
        for i in range(len(coords)):
            sorted_dists = np.sort(dists[i, :])
            k_distances[i] = sorted_dists[min(k, len(sorted_dists) - 1)]

        return k_distances

    def compute_celltype_mixing(
        self,
        type1: str,
        type2: str,
        radius: float,
    ) -> float:
        """
        Compute spatial mixing score between two cell types.

        Higher values indicate more mixing/intermixing.

        Parameters
        ----------
        type1 : str
            First cell type.
        type2 : str
            Second cell type.
        radius : float
            Neighborhood radius.

        Returns
        -------
        float
            Mixing score (0-1).
        """
        if self.data.cell_types is None:
            raise ValueError("Data does not have cell type annotations")

        mask1 = self.data.cell_types == type1
        mask2 = self.data.cell_types == type2

        coords1 = self.data.coords[mask1]
        coords2 = self.data.coords[mask2]

        if len(coords1) == 0 or len(coords2) == 0:
            return 0.0

        # Compute cross-distances
        cross_dists = distance.cdist(coords1, coords2)

        # Count type2 neighbors within radius for each type1 cell
        n_neighbors = np.sum(cross_dists <= radius, axis=1)

        # Normalize by expected count under random mixing
        expected = len(coords2) * (np.pi * radius**2) / self.domain_area
        mixing_score = np.mean(n_neighbors) / max(expected, 1e-10)

        return min(mixing_score, 1.0)

    def generate_similar_layout(
        self,
        n_cells: int,
        sender_fraction: float,
        receiver_fraction: float,
        seed: int = 42,
    ) -> SpatialLayout:
        """
        Generate a new layout similar to real data distribution.

        Parameters
        ----------
        n_cells : int
            Number of cells in new layout.
        sender_fraction : float
            Fraction of cells that are senders.
        receiver_fraction : float
            Fraction of cells that are receivers.
        seed : int
            Random seed.

        Returns
        -------
        SpatialLayout
            Generated spatial layout.
        """
        rng = np.random.default_rng(seed)

        # Generate positions using same density pattern
        # Sample from uniform disk with same radius
        r = self.max_radius * np.sqrt(rng.uniform(0, 1, n_cells))
        theta = rng.uniform(0, 2 * np.pi, n_cells)

        coords = np.column_stack(
            [
                self.center[0] + r * np.cos(theta),
                self.center[1] + r * np.sin(theta),
            ]
        )

        # Assign cell types
        n_senders = int(n_cells * sender_fraction)
        n_receivers = int(n_cells * receiver_fraction)
        n_other = n_cells - n_senders - n_receivers

        cell_types = np.array(
            ["sender"] * n_senders + ["receiver"] * n_receivers + ["other"] * n_other
        )
        rng.shuffle(cell_types)

        sender_indices = np.where(cell_types == "sender")[0]
        receiver_indices = np.where(cell_types == "receiver")[0]

        return SpatialLayout(
            coords=coords,
            cell_types=cell_types,
            sender_indices=sender_indices,
            receiver_indices=receiver_indices,
            domain_radius=self.max_radius,
            center=tuple(self.center),
            density=n_cells / self.domain_area,
        )

    def estimate_diffusion_length(
        self,
        sender_type: str,
        receiver_type: str,
        expression: np.ndarray,
        factor_gene_idx: int,
        response_gene_idx: int,
    ) -> float:
        """
        Estimate characteristic diffusion length from real data.

        Uses spatial decay of correlation to estimate lambda.

        Parameters
        ----------
        sender_type : str
            Sender cell type.
        receiver_type : str
            Receiver cell type.
        expression : np.ndarray
            Expression matrix (n_genes, n_cells).
        factor_gene_idx : int
            Index of factor gene.
        response_gene_idx : int
            Index of response gene.

        Returns
        -------
        float
            Estimated diffusion length in same units as coordinates.
        """
        if self.data.cell_types is None:
            raise ValueError("Data does not have cell type annotations")

        sender_mask = self.data.cell_types == sender_type
        receiver_mask = self.data.cell_types == receiver_type

        sender_coords = self.data.coords[sender_mask]
        receiver_coords = self.data.coords[receiver_mask]

        factor_expr = expression[factor_gene_idx, sender_mask]
        response_expr = expression[response_gene_idx, receiver_mask]

        # Compute distances from senders to receivers
        dists = distance.cdist(sender_coords, receiver_coords)

        # Weight by factor expression
        weights = factor_expr[:, np.newaxis] * np.ones((1, len(receiver_coords)))

        # Compute weighted average distance to responding cells
        response_threshold = np.percentile(response_expr, 75)
        responding_mask = response_expr > response_threshold

        if np.sum(responding_mask) == 0:
            return self.max_radius / 4  # Default estimate

        weighted_dists = dists[:, responding_mask] * weights[:, responding_mask]
        mean_dist = np.mean(weighted_dists)

        # Lambda is approximately the characteristic distance
        return mean_dist
