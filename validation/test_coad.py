#!/usr/bin/env python3
"""
Test sigdiscovPy with COAD CosMx high-plex dataset.

Dataset: coad_spatial.h5ad (16GB)
- High-plex CosMx spatial transcriptomics
"""

import numpy as np
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sigdiscovpy.core.normalization import standardize_matrix
from sigdiscovpy.core.weights import create_gaussian_weights
from sigdiscovpy.core.spatial_lag import compute_spatial_lag
from sigdiscovpy.core.metrics import compute_moran_from_lag, compute_ind_from_lag
from sigdiscovpy.analysis.pairwise_moran import pairwise_moran, pairwise_moran_directional
from sigdiscovpy.io.loaders import load_anndata


def load_coad_dataset():
    """Load COAD CosMx h5ad dataset."""
    data_path = Path("/Users/seongyongpark/project/sigdiscov/dataset/cosmx/coad_spatial.h5ad")

    print("Loading COAD CosMx dataset...")
    print(f"  File: {data_path}")
    print(f"  Size: 16 GB")

    start = time.time()
    data = load_anndata(str(data_path))
    load_time = time.time() - start

    print(f"  Load time: {load_time:.1f}s")
    print(f"  Expression: {data['expr'].shape[0]} genes x {data['expr'].shape[1]} cells")
    print(f"  Coordinates: {data['coords'].shape}")

    if data['cell_types'] is not None:
        unique_types = np.unique(data['cell_types'])
        print(f"  Cell types: {len(unique_types)} unique types")

    return data


def test_core_functions(data, use_gpu=False):
    """Test core functions on COAD data."""
    print(f"\n{'='*60}")
    print(f"Testing Core Functions (GPU={use_gpu})")
    print(f"{'='*60}")

    expr = data["expr"]
    coords = data["coords"]

    # Use subset for testing
    n_genes_test = 50
    n_cells_test = 10000

    np.random.seed(42)
    cell_idx = np.random.choice(expr.shape[1], min(n_cells_test, expr.shape[1]), replace=False)
    expr_sub = expr[:n_genes_test, cell_idx]
    coords_sub = coords[cell_idx]

    print(f"\nUsing subset: {n_genes_test} genes x {len(cell_idx)} cells")

    # 1. Standardization
    print("\n1. Testing standardize_matrix...")
    start = time.time()
    expr_norm = standardize_matrix(expr_sub, use_gpu=use_gpu)
    t_norm = time.time() - start
    print(f"   Time: {t_norm:.3f}s")
    print(f"   Mean: {expr_norm.mean(axis=1).mean():.2e} (should be ~0)")
    print(f"   Std: {expr_norm.std(axis=1).mean():.4f} (should be ~1)")

    # 2. Weight matrix
    print("\n2. Testing create_gaussian_weights...")
    radius = 100  # microns
    start = time.time()
    W = create_gaussian_weights(coords_sub, radius=radius, use_gpu=use_gpu)
    t_weights = time.time() - start
    W_dense = W.toarray() if hasattr(W, 'toarray') else W
    print(f"   Time: {t_weights:.3f}s")
    print(f"   Radius: {radius}")
    print(f"   Weight sum: {W_dense.sum():.2f}")
    print(f"   Non-zero: {(W_dense > 0).sum()}")
    print(f"   Sparsity: {100*(1 - (W_dense > 0).sum()/(W_dense.size)):.2f}%")

    # 3. Spatial lag
    print("\n3. Testing compute_spatial_lag...")
    start = time.time()
    lag = compute_spatial_lag(W, expr_norm[0], use_gpu=use_gpu)
    t_lag = time.time() - start
    print(f"   Time: {t_lag:.3f}s")
    print(f"   Lag range: [{lag.min():.4f}, {lag.max():.4f}]")

    # 4. Moran's I
    print("\n4. Testing compute_moran_from_lag...")
    z_f = expr_norm[0]
    lag_g = compute_spatial_lag(W, expr_norm[1], use_gpu=use_gpu)
    start = time.time()
    moran_i = compute_moran_from_lag(z_f, lag_g, use_gpu=use_gpu)
    t_moran = time.time() - start
    print(f"   Time: {t_moran:.6f}s")
    print(f"   Moran's I: {moran_i:.6f}")

    # 5. I_ND
    print("\n5. Testing compute_ind_from_lag...")
    start = time.time()
    ind = compute_ind_from_lag(z_f, lag_g, use_gpu=use_gpu)
    t_ind = time.time() - start
    print(f"   Time: {t_ind:.6f}s")
    print(f"   I_ND: {ind:.6f}")

    return {
        "n_cells": len(cell_idx),
        "weight_sum": W_dense.sum(),
        "moran_i": moran_i,
        "ind": ind,
    }


def test_pairwise_moran(data, use_gpu=False):
    """Test pairwise Moran's I."""
    print(f"\n{'='*60}")
    print(f"Testing Pairwise Moran's I (GPU={use_gpu})")
    print(f"{'='*60}")

    expr = data["expr"]
    coords = data["coords"]

    # Smaller subset for pairwise
    n_genes_test = 30
    n_cells_test = 5000

    np.random.seed(42)
    cell_idx = np.random.choice(expr.shape[1], min(n_cells_test, expr.shape[1]), replace=False)
    expr_sub = expr[:n_genes_test, cell_idx]
    coords_sub = coords[cell_idx]

    print(f"\nUsing subset: {n_genes_test} genes x {len(cell_idx)} cells")

    radius = 100
    print(f"Radius: {radius}")

    start = time.time()
    result = pairwise_moran(expr_sub, coords_sub, radius=radius, use_gpu=use_gpu)
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Shape: {result.shape}")
    print(f"  Diagonal: [{result.diagonal().min():.4f}, {result.diagonal().max():.4f}]")
    print(f"  Off-diag: [{np.min(result[~np.eye(n_genes_test, dtype=bool)]):.4f}, "
          f"{np.max(result[~np.eye(n_genes_test, dtype=bool)]):.4f}]")

    return result, elapsed


def test_celltype_analysis(data, use_gpu=False):
    """Test cell type analysis if cell types available."""
    print(f"\n{'='*60}")
    print(f"Testing Cell Type Analysis (GPU={use_gpu})")
    print(f"{'='*60}")

    if data['cell_types'] is None:
        print("No cell type annotations available, skipping.")
        return None, 0

    expr = data["expr"]
    coords = data["coords"]
    cell_types = data["cell_types"]

    # Get cell type distribution
    unique_types, counts = np.unique(cell_types, return_counts=True)
    print(f"\nCell type distribution (top 10):")
    for ct, count in sorted(zip(unique_types, counts), key=lambda x: -x[1])[:10]:
        print(f"  {ct}: {count:,} cells")

    # Pick two cell types
    sorted_types = [ct for ct, _ in sorted(zip(unique_types, counts), key=lambda x: -x[1])]
    sender_type = sorted_types[0]
    receiver_type = sorted_types[1] if len(sorted_types) > 1 else sorted_types[0]

    sender_idx = np.where(cell_types == sender_type)[0]
    receiver_idx = np.where(cell_types == receiver_type)[0]

    print(f"\nDirectional: {sender_type} -> {receiver_type}")
    print(f"  Senders: {len(sender_idx):,} cells")
    print(f"  Receivers: {len(receiver_idx):,} cells")

    # Subsample
    max_cells = 3000
    if len(sender_idx) > max_cells:
        sender_idx = np.random.choice(sender_idx, max_cells, replace=False)
    if len(receiver_idx) > max_cells:
        receiver_idx = np.random.choice(receiver_idx, max_cells, replace=False)

    n_genes_test = 20
    sender_expr = expr[:n_genes_test, sender_idx]
    receiver_expr = expr[:n_genes_test, receiver_idx]
    sender_coords = coords[sender_idx]
    receiver_coords = coords[receiver_idx]

    print(f"  Subsampled: {len(sender_idx)} senders, {len(receiver_idx)} receivers")
    print(f"  Genes: {n_genes_test}")

    radius = 100
    start = time.time()
    result = pairwise_moran_directional(
        sender_expr,
        receiver_expr,
        sender_coords,
        receiver_coords,
        radius=radius,
        use_gpu=use_gpu,
    )
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Shape: {result.shape}")
    print(f"  Range: [{result.min():.4f}, {result.max():.4f}]")

    return result, elapsed


def main():
    print("=" * 60)
    print("sigdiscovPy COAD CosMx High-Plex Validation")
    print("=" * 60)

    # Load data
    try:
        data = load_coad_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Test core functions
    results = test_core_functions(data, use_gpu=False)

    # Test pairwise Moran
    pairwise_result, pairwise_time = test_pairwise_moran(data, use_gpu=False)

    # Test cell type analysis
    celltype_result, celltype_time = test_celltype_analysis(data, use_gpu=False)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Dataset: COAD CosMx ({data['expr'].shape[1]:,} cells, {data['expr'].shape[0]} genes)")
    print(f"All tests completed successfully!")
    print(f"\nKey metrics:")
    print(f"  Weight sum (r=100): {results['weight_sum']:.2f}")
    print(f"  Sample Moran's I: {results['moran_i']:.6f}")
    print(f"  Sample I_ND: {results['ind']:.6f}")
    print(f"  Pairwise time (30 genes, 5k cells): {pairwise_time:.2f}s")
    if celltype_time > 0:
        print(f"  Cell type analysis time: {celltype_time:.2f}s")

    return 0


if __name__ == "__main__":
    exit(main())
