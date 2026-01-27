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
    SimulationConfig,
    DomainConfig,
    CellTypeConfig,
    PositionConfig,
    DiffusionConfig,
    ExpressionConfig,
    StochasticConfig,
    AnalysisConfig,
)
from sigdiscovpy.simulation.config.presets import SimulationPresets
from sigdiscovpy.simulation.runner import UnifiedSimulation

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
]
