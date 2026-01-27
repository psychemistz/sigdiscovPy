#!/usr/bin/env python3
"""
Macrophage → CAF TGFB1 Annular Analysis

Replicates the R analysis and compares results with:
results_macrophage_to_CAF_TGFB1_annular_20250818_150423.csv
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sigdiscovpy.core.normalization import standardize_matrix
from sigdiscovpy.core.weights import create_ring_weights, create_directional_weights
from sigdiscovpy.core.spatial_lag import compute_spatial_lag
from sigdiscovpy.core.metrics import compute_ind_from_lag
from sigdiscovpy.stats.fdr import apply_fdr_correction
from sigdiscovpy.io.loaders import load_anndata


# Annular ranges matching R analysis
ANNULAR_RANGES = [
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 50),
    (50, 100),
    (100, 200),
    (200, 300),
    (300, 500),
]


def load_coad_data():
    """Load COAD dataset and extract relevant cell types."""
    print("Loading COAD dataset...")
    data_path = "/Users/seongyongpark/project/sigdiscov/dataset/cosmx/coad_spatial.h5ad"

    data = load_anndata(data_path)

    expr = data['expr']  # genes x cells
    coords = data['coords']
    gene_names = data['gene_names']
    cell_types = data['cell_types']

    print(f"  Total: {expr.shape[0]} genes x {expr.shape[1]} cells")

    # Get macrophage and CAF indices
    macrophage_idx = np.where(cell_types == 'macrophage')[0]
    caf_idx = np.where(cell_types == 'CAF')[0]

    print(f"  Macrophages: {len(macrophage_idx)} cells")
    print(f"  CAF: {len(caf_idx)} cells")

    # Check if TGFB1 exists
    if 'TGFB1' not in gene_names:
        raise ValueError("TGFB1 not found in gene list")

    tgfb1_idx = gene_names.index('TGFB1')
    print(f"  TGFB1 gene index: {tgfb1_idx}")

    return {
        'expr': expr,
        'coords': coords,
        'gene_names': gene_names,
        'cell_types': cell_types,
        'macrophage_idx': macrophage_idx,
        'caf_idx': caf_idx,
        'tgfb1_idx': tgfb1_idx,
    }


def compute_annular_analysis(
    data: Dict,
    inner_radius: float,
    outer_radius: float,
    n_permutations: int = 999,
    use_gpu: bool = False,
) -> pd.DataFrame:
    """
    Compute annular spatial correlation for TGFB1 → all genes.

    Parameters
    ----------
    data : dict
        Dataset dictionary from load_coad_data()
    inner_radius : float
        Inner radius of annulus
    outer_radius : float
        Outer radius of annulus
    n_permutations : int
        Number of permutations for significance testing
    use_gpu : bool
        Whether to use GPU acceleration

    Returns
    -------
    pd.DataFrame
        Results with columns matching R output
    """
    expr = data['expr']
    coords = data['coords']
    gene_names = data['gene_names']
    macrophage_idx = data['macrophage_idx']
    caf_idx = data['caf_idx']
    tgfb1_idx = data['tgfb1_idx']

    # Extract sender (macrophage) and receiver (CAF) data
    sender_coords = coords[macrophage_idx]
    receiver_coords = coords[caf_idx]

    n_senders = len(macrophage_idx)
    n_receivers = len(caf_idx)
    n_genes = len(gene_names)

    # Global normalization of expression matrix
    expr_norm = standardize_matrix(expr, axis=1, use_gpu=use_gpu)

    # Get TGFB1 expression from senders (macrophages)
    tgfb1_expr = expr_norm[tgfb1_idx, macrophage_idx]

    # Get all gene expression from receivers (CAF)
    receiver_expr = expr_norm[:, caf_idx]  # (n_genes, n_receivers)

    # Create annular weight matrix
    sigma = outer_radius / 3.0
    W = create_directional_ring_weights(
        sender_coords,
        receiver_coords,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        sigma=sigma,
    )

    # Get weight sum and connection count
    W_dense = W.toarray() if hasattr(W, 'toarray') else W
    S0 = float(W_dense.sum())
    n_connections = int((W_dense > 0).sum())

    print(f"  Annulus {inner_radius}-{outer_radius}: S0={S0:.2f}, connections={n_connections}")

    if S0 < 1e-10:
        # No connections in this annulus
        return create_empty_results(gene_names, inner_radius, outer_radius, n_senders, n_receivers)

    # Compute spatial lag for all genes: W @ receiver_expr.T
    # W: (n_senders, n_receivers), receiver_expr.T: (n_receivers, n_genes)
    # lag_G: (n_senders, n_genes)
    lag_G = np.asarray(W @ receiver_expr.T)

    # Compute I_ND for all genes
    results = []

    # Vectorized computation of I_ND
    tgfb1_norm = np.linalg.norm(tgfb1_expr)
    lag_norms = np.linalg.norm(lag_G, axis=0)  # (n_genes,)

    # Dot products: tgfb1_expr @ lag_G
    dots = tgfb1_expr @ lag_G  # (n_genes,)

    # I_ND = dot / (||tgfb1|| * ||lag||)
    denominator = tgfb1_norm * lag_norms
    i_nd = np.where(denominator > 1e-10, dots / denominator, np.nan)

    # Permutation test
    if n_permutations > 0:
        perm_stats = np.zeros((n_permutations, n_genes))

        for p in range(n_permutations):
            # Permute TGFB1 expression
            perm_idx = np.random.permutation(n_senders)
            tgfb1_perm = tgfb1_expr[perm_idx]

            # Compute permuted I_ND
            perm_norm = np.linalg.norm(tgfb1_perm)
            perm_dots = tgfb1_perm @ lag_G
            perm_denom = perm_norm * lag_norms
            perm_stats[p] = np.where(perm_denom > 1e-10, perm_dots / perm_denom, np.nan)

        # Compute p-values (two-sided)
        perm_mean = np.nanmean(perm_stats, axis=0)
        perm_std = np.nanstd(perm_stats, axis=0)

        # Count exceedances
        exceedances = np.sum(np.abs(perm_stats) >= np.abs(i_nd), axis=0)
        pvalues = (exceedances + 1) / (n_permutations + 1)

        # Z-scores
        zscores = np.where(perm_std > 1e-10, (i_nd - perm_mean) / perm_std, 0.0)
    else:
        pvalues = np.full(n_genes, np.nan)
        zscores = np.full(n_genes, np.nan)
        perm_mean = np.full(n_genes, np.nan)
        perm_std = np.full(n_genes, np.nan)

    # Build results DataFrame
    for g in range(n_genes):
        results.append({
            'factor_gene': 'TGFB1',
            'target_gene': gene_names[g],
            'interaction': f'TGFB1→{gene_names[g]}',
            'morans_i_normalized': i_nd[g],
            'zscore': zscores[g],
            'pvalue': pvalues[g],
            'perm_mean': perm_mean[g] if n_permutations > 0 else np.nan,
            'perm_std': perm_std[g] if n_permutations > 0 else np.nan,
            'effect_size': zscores[g],
            'n_connections': n_connections,
            'S0': S0,
            'n_high_expr': n_senders,
            'n_total_senders': n_senders,
            'inner_radius': inner_radius,
            'annulus_range': f'{int(inner_radius)}-{int(outer_radius)}',
            'edge_type': 'circular',
            'sender_type': 'macrophage',
            'receiver_type': 'CAF',
            'radius': outer_radius,
            'weight_scheme': 'gaussian',
            'expr_norm': 'global',
            'min_expr_quantile': 0.0,
            'n_senders': n_senders,
            'n_receivers': n_receivers,
            'direction': 'macrophage→CAF',
            'analysis_mode': 'cross',
        })

    return pd.DataFrame(results)


def create_directional_ring_weights(
    sender_coords: np.ndarray,
    receiver_coords: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    sigma: float,
    row_normalize: bool = True,
) -> 'scipy.sparse.csr_matrix':
    """Create directional ring weights between senders and receivers."""
    from scipy import sparse as sp_sparse

    n_senders = sender_coords.shape[0]
    n_receivers = receiver_coords.shape[0]

    gaussian_factor = -1.0 / (2.0 * sigma * sigma)
    inner_sq = inner_radius * inner_radius
    outer_sq = outer_radius * outer_radius

    rows_list = []
    cols_list = []
    weights_list = []

    # Chunked computation for memory efficiency
    chunk_size = 1000

    for i_start in range(0, n_senders, chunk_size):
        i_end = min(i_start + chunk_size, n_senders)
        sender_chunk = sender_coords[i_start:i_end]

        # Compute distances
        diff_x = sender_chunk[:, 0:1] - receiver_coords[:, 0]
        diff_y = sender_chunk[:, 1:2] - receiver_coords[:, 1]
        dist_sq = diff_x**2 + diff_y**2

        # Find pairs in annulus
        mask = (dist_sq > inner_sq) & (dist_sq <= outer_sq)

        for local_i in range(i_end - i_start):
            global_i = i_start + local_i
            neighbors = np.where(mask[local_i])[0]

            if len(neighbors) > 0:
                dists = np.sqrt(dist_sq[local_i, neighbors])
                weights = np.exp(dists**2 * gaussian_factor)

                if row_normalize:
                    # Row normalize
                    row_sum = weights.sum()
                    if row_sum > 1e-10:
                        weights = weights / row_sum
                    else:
                        continue

                rows_list.extend([global_i] * len(neighbors))
                cols_list.extend(neighbors.tolist())
                weights_list.extend(weights.tolist())

    if len(rows_list) == 0:
        return sp_sparse.csr_matrix((n_senders, n_receivers), dtype=np.float64)

    return sp_sparse.csr_matrix(
        (weights_list, (rows_list, cols_list)),
        shape=(n_senders, n_receivers),
        dtype=np.float64,
    )


def create_empty_results(gene_names, inner_radius, outer_radius, n_senders, n_receivers):
    """Create empty results DataFrame when no connections exist."""
    results = []
    for gene in gene_names:
        results.append({
            'factor_gene': 'TGFB1',
            'target_gene': gene,
            'interaction': f'TGFB1→{gene}',
            'morans_i_normalized': np.nan,
            'zscore': np.nan,
            'pvalue': np.nan,
            'perm_mean': np.nan,
            'perm_std': np.nan,
            'effect_size': np.nan,
            'n_connections': 0,
            'S0': 0.0,
            'n_high_expr': n_senders,
            'n_total_senders': n_senders,
            'inner_radius': inner_radius,
            'annulus_range': f'{int(inner_radius)}-{int(outer_radius)}',
            'edge_type': 'circular',
            'sender_type': 'macrophage',
            'receiver_type': 'CAF',
            'radius': outer_radius,
            'weight_scheme': 'gaussian',
            'expr_norm': 'global',
            'min_expr_quantile': 0.0,
            'n_senders': n_senders,
            'n_receivers': n_receivers,
            'direction': 'macrophage→CAF',
            'analysis_mode': 'cross',
        })
    return pd.DataFrame(results)


def compare_with_r_results(py_results: pd.DataFrame, r_results_path: str):
    """Compare Python results with R reference results."""
    print("\n" + "="*60)
    print("Comparison with R Results")
    print("="*60)

    r_df = pd.read_csv(r_results_path)

    # Compare by annulus range
    for inner, outer in ANNULAR_RANGES:
        annulus_str = f'{inner}-{outer}'

        py_annulus = py_results[py_results['annulus_range'] == annulus_str].copy()
        r_annulus = r_df[r_df['annulus_range'] == annulus_str].copy()

        if len(py_annulus) == 0 or len(r_annulus) == 0:
            print(f"\n{annulus_str}: Missing data")
            continue

        # Merge on target gene
        merged = py_annulus.merge(
            r_annulus[['target_gene', 'morans_i_normalized', 'pvalue']],
            on='target_gene',
            suffixes=('_py', '_r'),
        )

        if len(merged) == 0:
            print(f"\n{annulus_str}: No matching genes")
            continue

        # Compute correlation
        valid_mask = ~(np.isnan(merged['morans_i_normalized_py']) | np.isnan(merged['morans_i_normalized_r']))
        valid_merged = merged[valid_mask]

        if len(valid_merged) > 0:
            corr = np.corrcoef(
                valid_merged['morans_i_normalized_py'],
                valid_merged['morans_i_normalized_r']
            )[0, 1]

            diff = np.abs(valid_merged['morans_i_normalized_py'] - valid_merged['morans_i_normalized_r'])
            max_diff = diff.max()
            mean_diff = diff.mean()

            print(f"\n{annulus_str}:")
            print(f"  Genes compared: {len(valid_merged)}")
            print(f"  Correlation: {corr:.10f}")
            print(f"  Mean diff: {mean_diff:.2e}")
            print(f"  Max diff: {max_diff:.2e}")
            print(f"  Status: {'PASS' if corr > 0.99 else 'CHECK'}")


def main():
    print("="*60)
    print("Macrophage → CAF TGFB1 Annular Analysis")
    print("="*60)

    # Load data
    start = time.time()
    data = load_coad_data()
    load_time = time.time() - start
    print(f"  Load time: {load_time:.1f}s")

    # Run analysis for each annulus
    all_results = []

    np.random.seed(42)  # For reproducibility

    for inner, outer in ANNULAR_RANGES:
        print(f"\nProcessing annulus {inner}-{outer}...")
        start = time.time()

        results = compute_annular_analysis(
            data,
            inner_radius=inner,
            outer_radius=outer,
            n_permutations=999,
            use_gpu=False,
        )

        elapsed = time.time() - start
        print(f"  Time: {elapsed:.1f}s")

        all_results.append(results)

    # Combine results
    combined = pd.concat(all_results, ignore_index=True)

    # Apply FDR correction
    fdr_result = apply_fdr_correction(combined['pvalue'].values, method='bh')
    combined['pvalue_fdr'] = fdr_result['adjusted_pvalues']
    combined['significant_fdr'] = fdr_result['significant']

    print(f"\nTotal results: {len(combined)} rows")
    print(f"Significant (FDR < 0.05): {combined['significant_fdr'].sum()}")

    # Save results
    output_path = Path(__file__).parent / "results_macrophage_CAF_TGFB1_python.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Compare with R results
    r_results_path = "/Users/seongyongpark/project/sigdiscov/output/cosmx/results_macrophage_to_CAF_TGFB1_annular_20250818_150423.csv"
    compare_with_r_results(combined, r_results_path)

    return 0


if __name__ == "__main__":
    exit(main())
