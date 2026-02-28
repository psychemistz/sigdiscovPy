"""
sigdiscovPy - GPU-accelerated spatial signature discovery for spatial transcriptomics

This package provides Python implementations of spatial correlation algorithms
with optional GPU acceleration via CuPy. It is designed to produce identical
outputs to the sigdiscov R package (>0.99 correlation).

Key Features:
- Bivariate Moran's I and I_ND (cosine similarity) metrics
- Pairwise Moran's I matrix computation
- GPU acceleration with automatic CPU fallback
- Genome-wide spatial interaction analysis
- Cell type pair analysis
- Permutation testing with FDR correction

Example:
    >>> from sigdiscovpy import pairwise_moran, create_weights
    >>> W = create_weights(coords, radius=100)
    >>> result = pairwise_moran(expr, W)
"""

__version__ = "0.1.0"
__author__ = "Seongyong Park"

# Core functions - lazy imports for faster startup
# Analysis functions
from sigdiscovpy.analysis.pairwise_moran import pairwise_moran
from sigdiscovpy.core.metrics import (
    compute_ind_from_lag,
    compute_metric_batch,
    compute_moran_from_lag,
)
from sigdiscovpy.core.normalization import standardize_matrix, standardize_vector
from sigdiscovpy.core.spatial_lag import compute_spatial_lag, compute_spatial_lag_batch
from sigdiscovpy.core.weights import (
    create_gaussian_weights,
    create_grid_weights,
    create_ring_weights,
    row_normalize_weights,
)

# GPU backend
from sigdiscovpy.gpu.backend import GPU_AVAILABLE, get_array_module

__all__ = [
    # Version
    "__version__",
    # Core - normalization
    "standardize_matrix",
    "standardize_vector",
    # Core - metrics
    "compute_moran_from_lag",
    "compute_ind_from_lag",
    "compute_metric_batch",
    # Core - weights
    "create_gaussian_weights",
    "create_grid_weights",
    "create_ring_weights",
    "row_normalize_weights",
    # Core - spatial lag
    "compute_spatial_lag",
    "compute_spatial_lag_batch",
    # Analysis
    "pairwise_moran",
    # GPU
    "get_array_module",
    "GPU_AVAILABLE",
]
