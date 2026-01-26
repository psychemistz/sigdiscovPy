"""
Genome-wide spatial interaction analysis.

Port of R genomewide_analysis() with GPU acceleration.
"""

from typing import Optional, List, Dict, Any, Literal
import numpy as np
from pathlib import Path
from sigdiscovpy.gpu.backend import get_array_module, GPU_AVAILABLE, ensure_numpy
from sigdiscovpy.core.normalization import standardize_matrix
from sigdiscovpy.core.weights import create_gaussian_weights, create_ring_weights
from sigdiscovpy.core.spatial_lag import compute_spatial_lag_batch
from sigdiscovpy.core.metrics import compute_metric_batch


def genomewide_analysis(
    expr,
    coords,
    factor_genes: List[str],
    gene_names: List[str],
    radii: List[float] = [10, 20, 30, 50, 100, 200, 300, 500],
    metric: Literal["moran", "ind"] = "ind",
    use_annular: bool = False,
    annular_width: float = 20.0,
    min_expr_quantile: float = 0.25,
    use_gpu: bool = True,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Genome-wide spatial interaction analysis.

    Computes spatial correlation metrics for factor genes against all target genes
    across multiple radii.

    Parameters
    ----------
    expr : array-like
        Expression matrix (genes x cells).
    coords : array-like
        Spatial coordinates (cells x 2).
    factor_genes : list
        Names of factor genes to analyze.
    gene_names : list
        Names of all genes (rows of expr).
    radii : list, default=[10, 20, 30, 50, 100, 200, 300, 500]
        Radii for spatial analysis.
    metric : {"moran", "ind"}, default="ind"
        Spatial correlation metric.
    use_annular : bool, default=False
        Use annular (ring) weights instead of disk weights.
    annular_width : float, default=20.0
        Width of annular ring (only if use_annular=True).
    min_expr_quantile : float, default=0.25
        Minimum expression quantile for factor selection.
    use_gpu : bool, default=True
        Use GPU acceleration if available.
    output_path : str, optional
        Path to save HDF5 results.
    verbose : bool, default=True
        Print progress messages.

    Returns
    -------
    dict
        Dictionary with:
        - 'matrices': dict of matrices by radius {radius: (n_factors x n_genes)}
        - 'factor_genes': list of factor gene names
        - 'target_genes': list of target gene names
        - 'radii': list of radii used

    Examples
    --------
    >>> result = genomewide_analysis(
    ...     expr, coords,
    ...     factor_genes=['IFNG', 'TNF', 'IL6'],
    ...     gene_names=gene_list,
    ...     radii=[50, 100, 200],
    ... )
    >>> result['matrices'][100].shape  # (3, n_genes)
    """
    expr = np.asarray(expr, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)

    n_genes, n_cells = expr.shape

    if verbose:
        print(f"Genome-wide analysis: {n_genes} genes, {n_cells} cells")
        print(f"Factors: {len(factor_genes)}, Radii: {len(radii)}")

    # Find factor gene indices
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    factor_indices = []
    factor_names_found = []
    for fg in factor_genes:
        if fg in gene_to_idx:
            factor_indices.append(gene_to_idx[fg])
            factor_names_found.append(fg)

    if not factor_indices:
        raise ValueError("No factor genes found in gene_names")

    if verbose:
        print(f"Found {len(factor_indices)}/{len(factor_genes)} factors")

    # Global normalization (CRITICAL)
    if verbose:
        print("Standardizing expression matrix (global normalization)...")
    expr_z = standardize_matrix(expr, axis=1, use_gpu=use_gpu)

    # Extract factor expression
    factor_expr = expr_z[factor_indices, :]

    results = {
        "matrices": {},
        "factor_genes": factor_names_found,
        "target_genes": gene_names,
        "radii": radii,
    }

    # Process each radius
    for r_idx, radius in enumerate(radii):
        if verbose:
            print(f"Processing radius {radius} ({r_idx + 1}/{len(radii)})...")

        # Create weight matrix
        if use_annular and r_idx > 0:
            inner_radius = max(0, radius - annular_width)
            W = create_ring_weights(
                coords,
                outer_radius=radius,
                inner_radius=inner_radius,
                row_normalize=True,
                use_gpu=use_gpu,
            )
        else:
            W = create_gaussian_weights(
                coords,
                radius=radius,
                row_normalize=True,
                use_gpu=use_gpu,
            )

        # Compute spatial lags for all genes: (n_cells, n_genes)
        lag_G = compute_spatial_lag_batch(W, expr_z.T, use_gpu=use_gpu)

        # Compute metrics for each factor
        n_factors = len(factor_indices)
        metrics_matrix = np.zeros((n_factors, n_genes), dtype=np.float64)

        for f_idx in range(n_factors):
            z_f = factor_expr[f_idx, :]
            metrics = compute_metric_batch(z_f, lag_G, metric=metric, use_gpu=use_gpu)
            metrics_matrix[f_idx, :] = metrics

        results["matrices"][radius] = metrics_matrix

    # Save to HDF5 if output path provided
    if output_path is not None:
        from sigdiscovpy.io.hdf5 import save_results_hdf5

        # Stack matrices along new axis: (n_radii, n_factors, n_genes)
        stacked = np.stack([results["matrices"][r] for r in radii], axis=0)
        save_results_hdf5(
            output_path,
            matrices={"genomewide_metrics": stacked},
            gene_names=gene_names,
            radii=radii,
            metadata={
                "n_factors": len(factor_names_found),
                "n_genes": n_genes,
                "metric": metric,
            },
        )
        if verbose:
            print(f"Results saved to {output_path}")

    return results


def extract_top_delta_i(
    metrics_dict: Dict[float, np.ndarray],
    radii: List[float],
    gene_names: List[str],
    factor_idx: int = 0,
    top_n: int = 100,
    baseline_radius: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Extract top genes by delta I (change in metric across radii).

    Parameters
    ----------
    metrics_dict : dict
        Dictionary mapping radius to metrics matrix.
    radii : list
        List of radii.
    gene_names : list
        Gene names.
    factor_idx : int, default=0
        Index of factor to analyze.
    top_n : int, default=100
        Number of top genes to return.
    baseline_radius : float, optional
        Radius to use as baseline. Default: smallest radius.

    Returns
    -------
    dict
        Dictionary with:
        - 'genes': Top gene names
        - 'delta_i': Delta I values
        - 'baseline': Baseline values
        - 'peak': Peak values
    """
    if baseline_radius is None:
        baseline_radius = min(radii)

    baseline = metrics_dict[baseline_radius][factor_idx, :]

    # Find peak metric across radii
    all_metrics = np.stack([metrics_dict[r][factor_idx, :] for r in radii], axis=0)
    peak = np.nanmax(all_metrics, axis=0)

    delta_i = peak - baseline

    # Get top genes
    sorted_idx = np.argsort(delta_i)[::-1][:top_n]

    return {
        "genes": [gene_names[i] for i in sorted_idx],
        "delta_i": delta_i[sorted_idx],
        "baseline": baseline[sorted_idx],
        "peak": peak[sorted_idx],
    }
