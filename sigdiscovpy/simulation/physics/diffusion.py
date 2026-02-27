"""
Diffusion model for simulating secreted factor concentration fields.

Uses Green's function solution for steady-state 2D diffusion with uptake:
    ∂c/∂t = D∇²c - k*c + S(r)

At steady state (∂c/∂t = 0), the solution for a point source is:
    c(r) = (A / 4πD) * K₀(r/λ)

where:
    - λ = √(D/k) is the characteristic diffusion length
    - K₀ is the modified Bessel function of the second kind
    - A is the source strength

For computational efficiency, we use the asymptotic approximation:
    c(r) ≈ A * exp(-r/λ) / √r
"""

from typing import Optional

import numpy as np

from sigdiscovpy.simulation.config.dataclasses import DiffusionConfig


class DiffusionSolver:
    """
    Solves steady-state diffusion equation for concentration fields.

    The model represents secreted factors diffusing from sender cells
    and being taken up by receiver cells, reaching a steady-state
    concentration profile.

    Parameters
    ----------
    config : DiffusionConfig
        Diffusion parameters (D, k_max, Kd, secretion_rate, active_threshold).

    Example
    -------
    >>> config = DiffusionConfig(D=100.0, k_max=10.0)
    >>> solver = DiffusionSolver(config)
    >>> concentrations, lambda_val = solver.solve(
    ...     sender_positions, sender_expression, cell_positions, n_density
    ... )
    """

    def __init__(self, config: DiffusionConfig):
        self.config = config

    def calculate_lambda(self, n_receivers: float, p_r: float = 1.0) -> float:
        """
        Calculate characteristic diffusion length.

        λ = √(D * Kd / (n_eff * k_max))

        Parameters
        ----------
        n_receivers : float
            Receiver cell density (cells per area).
        p_r : float
            Probability of receiver responding (effective uptake).

        Returns
        -------
        float
            Characteristic diffusion length in microns.
        """
        n_eff = n_receivers * p_r
        if n_eff <= 0:
            return np.inf

        return np.sqrt(self.config.D * self.config.Kd / (n_eff * self.config.k_max))

    def solve(
        self,
        sender_positions: np.ndarray,
        sender_expression: np.ndarray,
        cell_positions: np.ndarray,
        n_receivers_density: float,
        position_dict: Optional[dict[str, np.ndarray]] = None,
        p_r: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """
        Solve concentration field using superposition of point sources.

        Parameters
        ----------
        sender_positions : np.ndarray
            Sender cell coordinates (n_senders, 2).
        sender_expression : np.ndarray
            Factor expression levels (n_senders,).
        cell_positions : np.ndarray
            All cell coordinates (n_cells, 2).
        n_receivers_density : float
            Receiver cell density for lambda calculation.
        position_dict : Dict[str, np.ndarray], optional
            Position dictionary for grouping senders by location.
        p_r : float
            Probability of receiver responding.

        Returns
        -------
        concentrations : np.ndarray
            Concentration at each cell (n_cells,).
        lambda_val : float
            Characteristic diffusion length used.
        """
        lambda_val = self.calculate_lambda(n_receivers_density, p_r)
        n_cells = len(cell_positions)
        concentrations = np.zeros(n_cells)

        # Identify active senders (expression > threshold)
        active_threshold = self.config.active_threshold
        active_mask = sender_expression > active_threshold
        active_expr = sender_expression[active_mask]
        active_pos = sender_positions[active_mask]

        if len(active_pos) == 0:
            return concentrations, lambda_val

        # Compute concentration field
        if position_dict is not None:
            # Group senders by position for efficiency
            concentrations = self._solve_grouped(
                active_pos, active_expr, cell_positions, lambda_val
            )
        else:
            # Simple superposition
            concentrations = self._solve_superposition(
                active_pos, active_expr, cell_positions, lambda_val
            )

        return concentrations, lambda_val

    def _solve_superposition(
        self,
        source_positions: np.ndarray,
        source_strengths: np.ndarray,
        cell_positions: np.ndarray,
        lambda_val: float,
    ) -> np.ndarray:
        """
        Compute concentration using superposition of point sources.

        Matches reference core.py singularity handling:
        if r < 1e-3: C = Q * 100, else: C = Q * exp(-r/lambda) / sqrt(r)
        """
        n_cells = len(cell_positions)
        concentrations = np.zeros(n_cells)

        for _i, (pos, strength) in enumerate(zip(source_positions, source_strengths)):
            source_factor = strength * self.config.secretion_rate

            # Vectorized distance computation
            distances = np.linalg.norm(cell_positions - pos, axis=1)

            # Reference singularity handling: near-source cap
            near_mask = distances < 1e-3
            far_mask = ~near_mask

            if np.isfinite(lambda_val):
                contrib = np.zeros(n_cells)
                contrib[near_mask] = source_factor * 100
                contrib[far_mask] = (
                    source_factor
                    * np.exp(-distances[far_mask] / lambda_val)
                    / np.sqrt(distances[far_mask])
                )
            else:
                contrib = np.zeros(n_cells)
                contrib[near_mask] = source_factor * 100
                contrib[far_mask] = source_factor / distances[far_mask]

            concentrations += contrib

        return concentrations

    def _solve_grouped(
        self,
        active_pos: np.ndarray,
        active_expr: np.ndarray,
        cell_positions: np.ndarray,
        lambda_val: float,
    ) -> np.ndarray:
        """
        Compute concentration by grouping senders at same position.

        More efficient when multiple senders are at the same location.
        Matches reference core.py singularity handling.
        """
        n_cells = len(cell_positions)
        concentrations = np.zeros(n_cells)

        # Group by unique positions
        unique_positions = {}
        for pos, expr in zip(active_pos, active_expr):
            pos_key = tuple(pos)
            if pos_key not in unique_positions:
                unique_positions[pos_key] = {"pos": pos, "total_expr": 0.0}
            unique_positions[pos_key]["total_expr"] += expr

        # Compute contribution from each unique position
        for pos_data in unique_positions.values():
            source_pos = pos_data["pos"]
            total_factor = pos_data["total_expr"] * self.config.secretion_rate

            distances = np.linalg.norm(cell_positions - source_pos, axis=1)

            # Reference singularity handling
            near_mask = distances < 1e-3
            far_mask = ~near_mask

            if np.isfinite(lambda_val):
                contrib = np.zeros(n_cells)
                contrib[near_mask] = total_factor * 100
                contrib[far_mask] = (
                    total_factor
                    * np.exp(-distances[far_mask] / lambda_val)
                    / np.sqrt(distances[far_mask])
                )
            else:
                contrib = np.zeros(n_cells)
                contrib[near_mask] = total_factor * 100
                contrib[far_mask] = total_factor / distances[far_mask]

            concentrations += contrib

        return concentrations

    def solve_analytical(
        self,
        source_position: np.ndarray,
        source_strength: float,
        cell_positions: np.ndarray,
        lambda_val: float,
    ) -> np.ndarray:
        """
        Compute concentration from a single source analytically.

        This is the pure Green's function solution without any
        approximations, useful for validation.

        Parameters
        ----------
        source_position : np.ndarray
            Source coordinates (2,).
        source_strength : float
            Source strength (expression level).
        cell_positions : np.ndarray
            Target coordinates (n_cells, 2).
        lambda_val : float
            Characteristic diffusion length.

        Returns
        -------
        np.ndarray
            Concentration at each target position.
        """
        from scipy.special import kv  # Modified Bessel function K_ν

        distances = np.linalg.norm(cell_positions - source_position, axis=1)

        # Avoid singularity at source
        distances = np.maximum(distances, 1e-6)

        # Full Green's function: (A / 4πD) * K₀(r/λ)
        A = source_strength * self.config.secretion_rate
        prefactor = A / (4 * np.pi * self.config.D)

        if np.isfinite(lambda_val) and lambda_val > 0:
            concentrations = prefactor * kv(0, distances / lambda_val)
        else:
            # No uptake: logarithmic decay in 2D
            concentrations = prefactor * np.log(1 / distances)

        return concentrations
