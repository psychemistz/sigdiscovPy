"""Analysis modules for spatial signature discovery."""

from sigdiscovpy.analysis.celltype_pair import compute_celltype_pair_analysis, extract_top_pairs
from sigdiscovpy.analysis.delta_i import (
    classify_interaction_type,
    compute_delta_i,
    compute_delta_i_profile,
    find_peak_radius,
)
from sigdiscovpy.analysis.genomewide import extract_top_delta_i, genomewide_analysis
from sigdiscovpy.analysis.pairwise_moran import pairwise_moran, pairwise_moran_directional
from sigdiscovpy.analysis.roc_pr import (
    compute_gene_statistics,
    compute_roc_pr_curves,
    evaluate_all_directions,
    evaluate_condition,
    prepare_ground_truth,
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
    "compute_gene_statistics",
    "prepare_ground_truth",
    "evaluate_condition",
    "evaluate_all_directions",
    "compute_roc_pr_curves",
]
