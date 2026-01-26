"""Analysis modules for spatial signature discovery."""

from sigdiscovpy.analysis.pairwise_moran import pairwise_moran, pairwise_moran_directional
from sigdiscovpy.analysis.genomewide import genomewide_analysis, extract_top_delta_i
from sigdiscovpy.analysis.celltype_pair import compute_celltype_pair_analysis, extract_top_pairs
from sigdiscovpy.analysis.delta_i import (
    compute_delta_i,
    compute_delta_i_profile,
    find_peak_radius,
    classify_interaction_type,
)

__all__ = [
    "pairwise_moran",
    "pairwise_moran_directional",
    "genomewide_analysis",
    "extract_top_delta_i",
    "compute_celltype_pair_analysis",
    "extract_top_pairs",
    "compute_delta_i",
    "compute_delta_i_profile",
    "find_peak_radius",
    "classify_interaction_type",
]
