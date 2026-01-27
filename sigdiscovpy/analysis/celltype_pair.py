"""
Cell type pair spatial interaction analysis.

Analyzes spatial interactions between all sender-receiver cell type pairs.
"""

from typing import Any

import numpy as np

from sigdiscovpy.core.metrics import compute_metrics_matrix
from sigdiscovpy.core.normalization import standardize_matrix
from sigdiscovpy.core.spatial_lag import compute_spatial_lag_batch
from sigdiscovpy.core.weights import create_directional_weights


def compute_celltype_pair_analysis(
    expr,
    coords,
    cell_types: np.ndarray,
    factor_genes: list[str],
    gene_names: list[str],
    radii: list[float] = None,
    metric: str = "ind",
    min_cells: int = 10,
    use_gpu: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compute spatial metrics for all cell type pairs.

    Parameters
    ----------
    expr : array-like
        Expression matrix (genes x cells).
    coords : array-like
        Spatial coordinates (cells x 2).
    cell_types : array-like
        Cell type labels for each cell.
    factor_genes : list
        Names of factor genes.
    gene_names : list
        Names of all genes.
    radii : list, default=[50, 100, 200]
        Radii for spatial analysis.
    metric : str, default="ind"
        Spatial metric ("moran" or "ind").
    min_cells : int, default=10
        Minimum cells per cell type.
    use_gpu : bool, default=True
        Use GPU acceleration.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    dict
        Dictionary with:
        - 'matrices': dict of {(sender, receiver, radius): matrix}
        - 'pairs': list of (sender, receiver) tuples
        - 'radii': list of radii
        - 'factor_genes': factor gene names
        - 'target_genes': all gene names

    Examples
    --------
    >>> result = compute_celltype_pair_analysis(
    ...     expr, coords, cell_types,
    ...     factor_genes=['IFNG', 'TNF'],
    ...     gene_names=gene_list,
    ... )
    >>> key = ('T cell', 'Macrophage', 100)
    >>> result['matrices'][key].shape  # (n_factors, n_genes)
    """
    if radii is None:
        radii = [50, 100, 200]
    expr = np.asarray(expr, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    cell_types = np.asarray(cell_types)

    n_genes, n_cells = expr.shape

    # Get unique cell types with sufficient cells
    unique_types = np.unique(cell_types)
    valid_types = []
    for ct in unique_types:
        if np.sum(cell_types == ct) >= min_cells:
            valid_types.append(ct)

    if verbose:
        print(f"Cell types: {len(valid_types)} (with >= {min_cells} cells)")

    # Generate cell type pairs (sender -> receiver)
    pairs = []
    for sender in valid_types:
        for receiver in valid_types:
            if sender != receiver:
                pairs.append((sender, receiver))

    if verbose:
        print(f"Cell type pairs: {len(pairs)}")

    # Find factor indices
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    factor_indices = [gene_to_idx[g] for g in factor_genes if g in gene_to_idx]
    factor_names_found = [g for g in factor_genes if g in gene_to_idx]

    if not factor_indices:
        raise ValueError("No factor genes found")

    # Global normalization (CRITICAL)
    if verbose:
        print("Standardizing expression matrix (global normalization)...")
    expr_z = standardize_matrix(expr, axis=1, use_gpu=use_gpu)

    results = {
        "matrices": {},
        "pairs": pairs,
        "radii": radii,
        "factor_genes": factor_names_found,
        "target_genes": gene_names,
    }

    # Cache cell type indices and coordinates
    ct_data = {}
    for ct in valid_types:
        mask = cell_types == ct
        ct_data[ct] = {
            "indices": np.where(mask)[0],
            "coords": coords[mask],
            "n_cells": mask.sum(),
        }

    total_computations = len(pairs) * len(radii)
    computed = 0

    for sender, receiver in pairs:
        sender_info = ct_data[sender]
        receiver_info = ct_data[receiver]

        sender_expr = expr_z[factor_indices][:, sender_info["indices"]]
        receiver_expr = expr_z[:, receiver_info["indices"]]

        for radius in radii:
            # Create directional weight matrix (sender -> receiver)
            W = create_directional_weights(
                sender_info["coords"],
                receiver_info["coords"],
                radius=radius,
                row_normalize=True,
                use_gpu=use_gpu,
            )

            # Compute spatial lags: (n_sender_cells, n_genes)
            lag_G = compute_spatial_lag_batch(W, receiver_expr.T, use_gpu=use_gpu)

            # Compute metrics: (n_factors, n_genes)
            metrics = compute_metrics_matrix(
                sender_expr.T,  # (n_sender_cells, n_factors)
                lag_G,  # (n_sender_cells, n_genes)
                metric=metric,
                use_gpu=use_gpu,
            )

            results["matrices"][(sender, receiver, radius)] = metrics

            computed += 1
            if verbose and computed % 10 == 0:
                print(f"Progress: {computed}/{total_computations}")

    if verbose:
        print(f"Completed: {len(results['matrices'])} matrices")

    return results


def extract_top_pairs(
    results: dict[str, Any],
    factor_idx: int = 0,
    gene_idx: int = 0,
    top_n: int = 10,
) -> list[tuple[str, str, float, float]]:
    """
    Extract top cell type pairs by metric value.

    Parameters
    ----------
    results : dict
        Output from compute_celltype_pair_analysis.
    factor_idx : int, default=0
        Factor gene index.
    gene_idx : int, default=0
        Target gene index.
    top_n : int, default=10
        Number of top pairs.

    Returns
    -------
    list
        List of (sender, receiver, radius, metric_value) tuples.
    """
    pairs_metrics = []

    for (sender, receiver, radius), matrix in results["matrices"].items():
        value = matrix[factor_idx, gene_idx]
        if not np.isnan(value):
            pairs_metrics.append((sender, receiver, radius, value))

    # Sort by absolute metric value
    pairs_metrics.sort(key=lambda x: abs(x[3]), reverse=True)

    return pairs_metrics[:top_n]
