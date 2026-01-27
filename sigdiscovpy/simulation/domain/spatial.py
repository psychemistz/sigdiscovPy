"""
Spatial domain generation for simulation.

Provides classes for:
- Generating cell positions in a circular domain
- Positioning sender cells (center, fixed, random)
- Distributing senders across multiple positions
"""

import numpy as np

from sigdiscovpy.simulation.config.dataclasses import (
    DomainConfig,
    PositionConfig,
    SenderPositionMode,
)


class SpatialDomain:
    """
    Generates cell positions in a circular spatial domain.

    Cells are uniformly distributed within a circular region, which is
    the standard geometry for spatial transcriptomics tissue sections.

    Parameters
    ----------
    config : DomainConfig
        Domain configuration specifying n_cells, max_radius, center, and seed.

    Example
    -------
    >>> config = DomainConfig(n_cells=10000, max_radius=1000.0)
    >>> domain = SpatialDomain(config)
    >>> positions = domain.generate_positions()
    >>> positions.shape
    (10000, 2)
    """

    def __init__(self, config: DomainConfig):
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)

    def generate_positions(self) -> np.ndarray:
        """
        Generate uniformly distributed positions in circular domain.

        Uses inverse transform sampling to ensure uniform distribution
        within the circle (not just uniform angles and radii).

        Returns
        -------
        np.ndarray
            Cell positions of shape (n_cells, 2).
        """
        n = self.config.n_cells
        center = np.array(self.config.center)
        max_r = self.config.max_radius

        # Uniform distribution in circle: r = sqrt(U) * max_r
        angles = self._rng.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(self._rng.uniform(0, 1, n)) * max_r

        positions = np.column_stack(
            [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)]
        )

        return positions

    def get_domain_area(self) -> float:
        """Get the area of the circular domain."""
        return np.pi * self.config.max_radius**2


class SenderPositionGenerator:
    """
    Generates sender cell positions based on configuration.

    Supports three modes:
    - CENTER: All senders at the domain center
    - FIXED_5: Senders at 5 fixed positions (Center, N, S, E, W)
    - RANDOM: Senders at random positions with minimum separation

    Parameters
    ----------
    config : PositionConfig
        Position configuration specifying mode and parameters.
    domain_config : DomainConfig
        Domain configuration for center and size reference.

    Example
    -------
    >>> pos_config = PositionConfig(mode=SenderPositionMode.FIXED_5)
    >>> dom_config = DomainConfig()
    >>> generator = SenderPositionGenerator(pos_config, dom_config)
    >>> positions = generator.generate_positions()
    >>> list(positions.keys())
    ['C', 'W', 'E', 'N', 'S']
    """

    def __init__(self, config: PositionConfig, domain_config: DomainConfig):
        self.config = config
        self.domain_config = domain_config
        self.center = np.array(domain_config.center)
        self._rng = np.random.default_rng(domain_config.random_seed)

    def generate_positions(self) -> dict[str, np.ndarray]:
        """
        Generate sender position dictionary.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping position labels to coordinates.
        """
        if self.config.mode == SenderPositionMode.CENTER:
            return {"C": self.center.copy()}

        elif self.config.mode == SenderPositionMode.FIXED_5:
            return self._generate_fixed_5()

        elif self.config.mode == SenderPositionMode.RANDOM:
            return self._generate_random_positions()

        else:
            raise ValueError(f"Unknown position mode: {self.config.mode}")

    def _generate_fixed_5(self) -> dict[str, np.ndarray]:
        """Generate 5 fixed positions: Center, West, East, North, South."""
        offset = self.config.offset_distance
        return {
            "C": self.center.copy(),
            "W": np.array([self.center[0] - offset, self.center[1]]),
            "E": np.array([self.center[0] + offset, self.center[1]]),
            "N": np.array([self.center[0], self.center[1] + offset]),
            "S": np.array([self.center[0], self.center[1] - offset]),
        }

    def _generate_random_positions(self) -> dict[str, np.ndarray]:
        """Generate random positions with minimum separation."""
        positions = {}
        coords_list = []
        effective_radius = self.domain_config.max_radius * 0.8

        attempts = 0
        max_attempts = 1000

        while len(positions) < self.config.n_positions and attempts < max_attempts:
            angle = self._rng.uniform(0, 2 * np.pi)
            r = np.sqrt(self._rng.uniform()) * effective_radius

            new_pos = np.array(
                [self.center[0] + r * np.cos(angle), self.center[1] + r * np.sin(angle)]
            )

            if len(coords_list) == 0:
                valid = True
            else:
                distances = [np.linalg.norm(new_pos - existing) for existing in coords_list]
                valid = min(distances) >= self.config.min_separation

            if valid:
                pos_label = f"P{len(positions) + 1}"
                positions[pos_label] = new_pos
                coords_list.append(new_pos)

            attempts += 1

        if len(positions) < self.config.n_positions:
            import warnings

            warnings.warn(
                f"Only placed {len(positions)}/{self.config.n_positions} positions "
                f"after {max_attempts} attempts. Consider reducing min_separation.",
                stacklevel=2,
            )

        return positions

    @staticmethod
    def distribute_senders(
        n_senders: int,
        position_dict: dict[str, np.ndarray],
        rng: np.random.Generator = None,
    ) -> tuple[dict[str, int], list[tuple[str, np.ndarray]]]:
        """
        Distribute sender cells across positions.

        Each position gets at least one sender, then remaining senders
        are distributed randomly.

        Parameters
        ----------
        n_senders : int
            Total number of senders to distribute.
        position_dict : Dict[str, np.ndarray]
            Dictionary of position labels to coordinates.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        assignments : Dict[str, int]
            Number of senders assigned to each position.
        sender_positions : List[Tuple[str, np.ndarray]]
            List of (position_label, coordinates) for each sender.

        Raises
        ------
        ValueError
            If n_senders < number of positions.
        """
        if rng is None:
            rng = np.random.default_rng()

        position_names = list(position_dict.keys())
        n_pos = len(position_names)

        if n_senders < n_pos:
            raise ValueError(f"n_senders ({n_senders}) must be >= number of positions ({n_pos})")

        # Each position gets at least one sender
        assignments = dict.fromkeys(position_names, 1)
        sender_positions = [(name, position_dict[name].copy()) for name in position_names]

        # Distribute remaining senders randomly
        remaining = n_senders - n_pos
        for _ in range(remaining):
            chosen = rng.choice(position_names)
            assignments[chosen] += 1
            sender_positions.append((chosen, position_dict[chosen].copy()))

        return assignments, sender_positions

    @staticmethod
    def assign_cells_to_positions(
        cell_positions: np.ndarray,
        sender_positions: list[tuple[str, np.ndarray]],
        cell_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Move selected cells to their assigned sender positions.

        Parameters
        ----------
        cell_positions : np.ndarray
            All cell positions (n_cells, 2). Modified in place.
        sender_positions : List[Tuple[str, np.ndarray]]
            List of (label, coordinates) for each sender.
        cell_indices : np.ndarray
            Indices of cells to move (one per sender position).

        Returns
        -------
        np.ndarray
            Modified cell_positions array.
        """
        for i, (_, coords) in enumerate(sender_positions):
            if i < len(cell_indices):
                cell_positions[cell_indices[i]] = coords

        return cell_positions
