"""
I_ND computation for simulation using existing sigdiscovpy core.

This module bridges the simulation framework with sigdiscovpy's core
spatial correlation functions, ensuring consistent computation between
simulated and real data analysis.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy import spatial

from sigdiscovpy.simulation.config.dataclasses import (
    AnalysisConfig,
    WeightType,
)

# Import core sigdiscovpy functions
from sigdiscovpy.core.normalization import standardize_matrix
from sigdiscovpy.core.weights import (
    create_gaussian_weights,
    create_ring_weights,
    row_normalize_weights,
)
from sigdiscovpy.core.spatial_lag import compute_spatial_lag
from sigdiscovpy.core.metrics import compute_ind_from_lag


class INDComputer:
    """
    Computes I_ND using existing sigdiscovpy core functions.

    This ensures that simulation I_ND values are computed identically
    to real data analysis, enabling valid comparisons.

    Parameters
    ----------
    config : AnalysisConfig
        Analysis configuration specifying radii, weights, etc.
    use_gpu : bool
        Whether to use GPU acceleration.

    Example
    -------
    >>> config = AnalysisConfig(radii=[100, 200, 300])
    >>> computer = INDComputer(config)
    >>> ind_values = computer.compute_at_radii(
    ...     sender_indices, receiver_indices, positions,
    ...     factor_expr, response_expr
    ... )
    """

    def __init__(self, config: AnalysisConfig, use_gpu: bool = False):
        self.config = config
        self.use_gpu = use_gpu

    def compute(
        self,
        sender_indices: np.ndarray,
        receiver_indices: np.ndarray,
        all_positions: np.ndarray,
        factor_expr: np.ndarray,
        response_expr: np.ndarray,
        radius: float,
    ) -> Tuple[float, int]:
        """
        Compute I_ND at a single radius using sigdiscovpy core.

        Workflow:
        1. Global normalization via standardize_matrix()
        2. Build weight matrix via create_*_weights()
        3. Row-normalize weights
        4. Compute spatial lag via compute_spatial_lag()
        5. Compute I_ND via compute_ind_from_lag()

        Parameters
        ----------
        sender_indices : np.ndarray
            Indices of sender cells.
        receiver_indices : np.ndarray
            Indices of receiver cells.
        all_positions : np.ndarray
            All cell positions (n_cells, 2).
        factor_expr : np.ndarray
            Factor expression for all cells (n_cells,).
        response_expr : np.ndarray
            Response expression for all cells (n_cells,).
        radius : float
            Radius for weight computation.

        Returns
        -------
        I_ND : float
            Normalized directional Moran's I value.
        n_connections : int
            Number of sender-receiver connections.
        """
        if self.config.use_sigdiscov_core:
            return self._compute_with_core(
                sender_indices, receiver_indices, all_positions,
                factor_expr, response_expr, radius
            )
        else:
            return self._compute_simple(
                sender_indices, receiver_indices, all_positions,
                factor_expr, response_expr, radius
            )

    def _compute_with_core(
        self,
        sender_indices: np.ndarray,
        receiver_indices: np.ndarray,
        all_positions: np.ndarray,
        factor_expr: np.ndarray,
        response_expr: np.ndarray,
        radius: float,
    ) -> Tuple[float, int]:
        """Compute I_ND using sigdiscovpy core functions."""
        # Step 1: Global normalization (CRITICAL)
        # Stack as 2 x n_cells matrix, normalize gene-wise
        expr_matrix = np.vstack([factor_expr, response_expr])
        expr_norm = standardize_matrix(expr_matrix, axis=1, use_gpu=self.use_gpu)

        # Extract normalized values
        z_factor_all = expr_norm[0, :]
        z_response_all = expr_norm[1, :]

        # Get sender/receiver subsets from globally normalized data
        z_f = z_factor_all[sender_indices]
        z_r = z_response_all[receiver_indices]

        sender_pos = all_positions[sender_indices]
        receiver_pos = all_positions[receiver_indices]

        # Step 2: Build weight matrix
        W = self._create_weight_matrix(sender_pos, receiver_pos, radius)
        n_connections = int(W.sum())

        if n_connections == 0:
            return 0.0, 0

        # Step 3: Row-normalize
        W_norm = row_normalize_weights(W)

        # Step 4: Compute spatial lag
        lag = compute_spatial_lag(W_norm, z_r, use_gpu=self.use_gpu)

        # Step 5: Compute I_ND
        I_ND = compute_ind_from_lag(z_f, lag, use_gpu=self.use_gpu)

        return I_ND, n_connections

    def _compute_simple(
        self,
        sender_indices: np.ndarray,
        receiver_indices: np.ndarray,
        all_positions: np.ndarray,
        factor_expr: np.ndarray,
        response_expr: np.ndarray,
        radius: float,
    ) -> Tuple[float, int]:
        """
        Simple I_ND computation (fallback/reference implementation).

        This matches the original unified_simulation.py implementation.
        """
        # Global normalization
        mu_f, sigma_f = np.mean(factor_expr), np.std(factor_expr)
        mu_r, sigma_r = np.mean(response_expr), np.std(response_expr)

        z_f = (factor_expr[sender_indices] - mu_f) / (sigma_f + 1e-10)
        z_r = (response_expr[receiver_indices] - mu_r) / (sigma_r + 1e-10)

        sender_pos = all_positions[sender_indices]
        receiver_pos = all_positions[receiver_indices]

        # Build weight matrix
        dists = spatial.distance.cdist(sender_pos, receiver_pos)
        W = self._build_simple_weight_matrix(dists, radius)

        # Row-normalize
        row_sums = W.sum(axis=1, keepdims=True)
        n_connections = int(W.sum())

        if n_connections == 0:
            return 0.0, 0

        row_sums[row_sums == 0] = 1.0
        W_norm = W / row_sums

        # Compute I_ND
        lag = W_norm @ z_r
        numerator = np.dot(z_f, lag)
        norm_f = np.linalg.norm(z_f)
        norm_lag = np.linalg.norm(lag)

        if norm_f > 1e-10 and norm_lag > 1e-10:
            I_ND = numerator / (norm_f * norm_lag)
        else:
            I_ND = 0.0

        return I_ND, n_connections

    def _create_weight_matrix(
        self,
        sender_pos: np.ndarray,
        receiver_pos: np.ndarray,
        radius: float,
    ):
        """Create weight matrix using sigdiscovpy core."""
        if self.config.weight_type == WeightType.RING:
            # For ring weights, use create_ring_weights with appropriate inner/outer
            half_bw = self.config.bandwidth / 2.0
            inner = max(0, radius - half_bw)
            outer = radius + half_bw

            # Create combined coordinates and get submatrix
            all_coords = np.vstack([sender_pos, receiver_pos])
            n_senders = len(sender_pos)

            W_full = create_ring_weights(
                all_coords,
                outer_radius=outer,
                inner_radius=inner,
                row_normalize=False,
                use_gpu=self.use_gpu,
            )

            # Extract sender-to-receiver submatrix
            W = W_full[:n_senders, n_senders:].toarray()
            return W

        elif self.config.weight_type == WeightType.ANNULAR:
            inner = max(0, radius - self.config.annular_width)
            outer = radius

            all_coords = np.vstack([sender_pos, receiver_pos])
            n_senders = len(sender_pos)

            W_full = create_ring_weights(
                all_coords,
                outer_radius=outer,
                inner_radius=inner,
                row_normalize=False,
                use_gpu=self.use_gpu,
            )

            W = W_full[:n_senders, n_senders:].toarray()
            return W

        elif self.config.weight_type == WeightType.GAUSSIAN:
            all_coords = np.vstack([sender_pos, receiver_pos])
            n_senders = len(sender_pos)

            W_full = create_gaussian_weights(
                all_coords,
                radius=radius,
                sigma=self.config.bandwidth,
                row_normalize=False,
                use_gpu=self.use_gpu,
            )

            W = W_full[:n_senders, n_senders:].toarray()
            return W

        else:
            raise ValueError(f"Unknown weight type: {self.config.weight_type}")

    def _build_simple_weight_matrix(
        self,
        dists: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Build weight matrix based on weight type (simple implementation)."""
        if self.config.weight_type == WeightType.RING:
            half_bw = self.config.bandwidth / 2.0
            lower = radius - half_bw
            upper = radius + half_bw
            return ((dists > lower) & (dists <= upper)).astype(float)

        elif self.config.weight_type == WeightType.ANNULAR:
            inner = max(0, radius - self.config.annular_width)
            outer = radius
            return ((dists > inner) & (dists <= outer)).astype(float)

        elif self.config.weight_type == WeightType.GAUSSIAN:
            sigma = self.config.bandwidth
            return np.exp(-0.5 * (dists / sigma) ** 2)

        else:
            raise ValueError(f"Unknown weight type: {self.config.weight_type}")

    def compute_at_radii(
        self,
        sender_indices: np.ndarray,
        receiver_indices: np.ndarray,
        all_positions: np.ndarray,
        factor_expr: np.ndarray,
        response_expr: np.ndarray,
        radii: Optional[List[float]] = None,
    ) -> List[dict]:
        """
        Compute I_ND at multiple radii.

        Parameters
        ----------
        sender_indices, receiver_indices, all_positions,
        factor_expr, response_expr : as in compute()
        radii : List[float], optional
            List of radii. If None, uses config.radii.

        Returns
        -------
        List[dict]
            List of dicts with 'radius', 'I_ND', 'n_connections'.
        """
        if radii is None:
            radii = self.config.radii

        results = []
        for radius in radii:
            i_nd, n_conn = self.compute(
                sender_indices, receiver_indices, all_positions,
                factor_expr, response_expr, radius
            )
            results.append({
                'radius': radius,
                'I_ND': i_nd,
                'n_connections': n_conn,
            })

        return results
