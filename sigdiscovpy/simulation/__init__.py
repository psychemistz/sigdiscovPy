"""
Simulation module for spatial transcriptomics analysis.

Provides tools for:
- Generating synthetic spatial data with known ground truth
- Simulating diffusion-based cell-cell communication
- Validating I_ND metric behavior under controlled conditions
- Benchmarking against real data patterns

Example:
    >>> from sigdiscovpy.simulation import UnifiedSimulation, SimulationPresets
    >>> config = SimulationPresets.default()
    >>> sim = UnifiedSimulation(config)
    >>> results = sim.run()
"""

from sigdiscovpy.simulation.config.dataclasses import (
    AnalysisConfig,
    CellTypeConfig,
    DiffusionConfig,
    DomainConfig,
    ExpressionConfig,
    PositionConfig,
    SimulationConfig,
    StochasticConfig,
)
from sigdiscovpy.simulation.config.presets import SimulationPresets

# Real data support
from sigdiscovpy.simulation.realdata import (
    SpatialData,
    SpatialLayout,
    SpatialLayoutGenerator,
    load_anndata,
    load_cosmx,
    load_csv,
)
from sigdiscovpy.simulation.runner import UnifiedSimulation

# Visualization
from sigdiscovpy.simulation.visualization import (
    plot_ind_curve,
    plot_simulation_summary,
    plot_spatial_cells,
)

__all__ = [
    # Config
    "SimulationConfig",
    "DomainConfig",
    "CellTypeConfig",
    "PositionConfig",
    "DiffusionConfig",
    "ExpressionConfig",
    "StochasticConfig",
    "AnalysisConfig",
    "SimulationPresets",
    # Runner
    "UnifiedSimulation",
    # Real data
    "SpatialData",
    "SpatialLayout",
    "SpatialLayoutGenerator",
    "load_anndata",
    "load_cosmx",
    "load_csv",
    # Visualization
    "plot_spatial_cells",
    "plot_ind_curve",
    "plot_simulation_summary",
]
