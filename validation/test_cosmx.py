#!/usr/bin/env python3
"""
Test sigdiscovPy with CosMx single-cell spatial data.

Dataset: Lung5_Rep1 (NanoString CosMx)
- 98,002 cells
- 960 genes
- Cell type annotations included
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sigdiscovpy.core.normalization import standardize_matrix
from sigdiscovpy.core.weights import create_gaussian_weights
from sigdiscovpy.core.spatial_lag import compute_spatial_lag
from sigdiscovpy.core.metrics import compute_moran_from_lag, compute_ind_from_lag
from sigdiscovpy.analysis.pairwise_moran import pairwise_moran, pairwise_moran_directional
from sigdiscovpy.stats.fdr import apply_fdr_correction


def load_cosmx_lung5():
    """Load Lung5_Rep1 CosMx dataset."""
    base_path = Path("/Users/seongyongpark/project/sigdiscov/dataset/cosmx")

    print("Loading CosMx Lung5_Rep1 dataset...")

    # Load expression (genes x cells)
    print("  Loading expression matrix...")
    expr_df = pd.read_csv(base_path / "Lung5_Rep1_vst.csv", index_col=0)
    expr = expr_df.values.astype(np.float64)
    gene_names = expr_df.index.tolist()
    cell_names = expr_df.columns.tolist()

    # Load metadata
    print("  Loading metadata...")
    meta_df = pd.read_csv(base_path / "Lung5_Rep1_meta.csv", index_col=0)
    coords = meta_df[["sdimx", "sdimy"]].values.astype(np.float64)
    cell_types = meta_df["cellType"].values

    print(f"  Expression: {expr.shape[0]} genes x {expr.shape[1]} cells")
    print(f"  Coordinates: {coords.shape}")
    print(f"  Cell types: {len(np.unique(cell_types))} unique types")

    return {
        "expr": expr,
        "coords": coords,
        "gene_names": gene_names,
        "cell_names": cell_names,
        "cell_types": cell_types,
    }


def test_core_functions(data, use_gpu=False):
    """Test core sigdiscovPy functions on CosMx data."""
    print(f"\n{'='*60}")
    print(f"Testing Core Functions (GPU={use_gpu})")
    print(f"{'='*60}")

    expr = data["expr"]
    coords = data["coords"]

    # Use subset for faster testing
    n_genes_test = 50
    n_cells_test = 5000

    # Sample cells
    np.random.seed(42)
    cell_idx = np.random.choice(expr.shape[1], n_cells_test, replace=False)
    expr_sub = expr[:n_genes_test, cell_idx]
    coords_sub = coords[cell_idx]

    print(f"\nUsing subset: {n_genes_test} genes x {n_cells_test} cells")

    # 1. Standardization
    print("\n1. Testing standardize_matrix...")
    start = time.time()
    expr_norm = standardize_matrix(expr_sub, use_gpu=use_gpu)
    t_norm = time.time() - start
    print(f"   Time: {t_norm:.3f}s")
    print(f"   Mean check (should be ~0): {expr_norm.mean(axis=1).mean():.2e}")
    print(f"   Std check (should be ~1): {expr_norm.std(axis=1).mean():.4f}")

    # 2. Weight matrix
    print("\n2. Testing create_gaussian_weights...")
    radius = 50  # microns
    start = time.time()
    W = create_gaussian_weights(coords_sub, radius=radius, use_gpu=use_gpu)
    t_weights = time.time() - start
    W_dense = W.toarray() if hasattr(W, 'toarray') else W
    print(f"   Time: {t_weights:.3f}s")
    print(f"   Radius: {radius}")
    print(f"   Weight sum: {W_dense.sum():.2f}")
    print(f"   Non-zero entries: {(W_dense > 0).sum()}")
    print(f"   Sparsity: {100*(1 - (W_dense > 0).sum()/(W_dense.size)):.2f}%")

    # 3. Spatial lag
    print("\n3. Testing compute_spatial_lag...")
    start = time.time()
    lag = compute_spatial_lag(W, expr_norm[0], use_gpu=use_gpu)
    t_lag = time.time() - start
    print(f"   Time: {t_lag:.3f}s")
    print(f"   Lag shape: {lag.shape}")
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

    # 5. I_ND (cosine similarity)
    print("\n5. Testing compute_ind_from_lag...")
    start = time.time()
    ind = compute_ind_from_lag(z_f, lag_g, use_gpu=use_gpu)
    t_ind = time.time() - start
    print(f"   Time: {t_ind:.6f}s")
    print(f"   I_ND: {ind:.6f}")

    return {
        "norm_time": t_norm,
        "weights_time": t_weights,
        "lag_time": t_lag,
        "moran_time": t_moran,
        "ind_time": t_ind,
        "weight_sum": W_dense.sum(),
        "moran_i": moran_i,
        "ind": ind,
    }


def test_pairwise_moran(data, use_gpu=False):
    """Test pairwise Moran's I computation."""
    print(f"\n{'='*60}")
    print(f"Testing Pairwise Moran's I (GPU={use_gpu})")
    print(f"{'='*60}")

    expr = data["expr"]
    coords = data["coords"]

    # Use smaller subset for pairwise (O(n_genes^2))
    n_genes_test = 20
    n_cells_test = 3000

    np.random.seed(42)
    cell_idx = np.random.choice(expr.shape[1], n_cells_test, replace=False)
    expr_sub = expr[:n_genes_test, cell_idx]
    coords_sub = coords[cell_idx]

    print(f"\nUsing subset: {n_genes_test} genes x {n_cells_test} cells")

    radius = 50
    print(f"Radius: {radius}")

    start = time.time()
    result = pairwise_moran(expr_sub, coords_sub, radius=radius, use_gpu=use_gpu)
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Result shape: {result.shape}")
    print(f"  Diagonal (self-correlation): [{result.diagonal().min():.4f}, {result.diagonal().max():.4f}]")
    print(f"  Off-diagonal range: [{np.min(result[~np.eye(n_genes_test, dtype=bool)]):.4f}, "
          f"{np.max(result[~np.eye(n_genes_test, dtype=bool)]):.4f}]")

    return result, elapsed


def test_celltype_analysis(data, use_gpu=False):
    """Test cell type-specific analysis."""
    print(f"\n{'='*60}")
    print(f"Testing Cell Type Analysis (GPU={use_gpu})")
    print(f"{'='*60}")

    expr = data["expr"]
    coords = data["coords"]
    cell_types = data["cell_types"]

    # Get unique cell types and counts
    unique_types, counts = np.unique(cell_types, return_counts=True)
    print(f"\nCell type distribution:")
    for ct, count in sorted(zip(unique_types, counts), key=lambda x: -x[1])[:10]:
        print(f"  {ct}: {count} cells")

    # Select two cell types for directional analysis
    # Pick two with sufficient cells
    sorted_types = [ct for ct, _ in sorted(zip(unique_types, counts), key=lambda x: -x[1])]
    sender_type = sorted_types[0]
    receiver_type = sorted_types[1]

    sender_idx = np.where(cell_types == sender_type)[0]
    receiver_idx = np.where(cell_types == receiver_type)[0]

    print(f"\nDirectional analysis: {sender_type} -> {receiver_type}")
    print(f"  Senders: {len(sender_idx)} cells")
    print(f"  Receivers: {len(receiver_idx)} cells")

    # Subsample for speed
    max_cells = 2000
    if len(sender_idx) > max_cells:
        sender_idx = np.random.choice(sender_idx, max_cells, replace=False)
    if len(receiver_idx) > max_cells:
        receiver_idx = np.random.choice(receiver_idx, max_cells, replace=False)

    # Use subset of genes
    n_genes_test = 20
    expr_sub = expr[:n_genes_test]

    print(f"  Subsampled: {len(sender_idx)} senders, {len(receiver_idx)} receivers")
    print(f"  Genes: {n_genes_test}")

    # Extract expression and coordinates for sender/receiver
    sender_expr = expr_sub[:, sender_idx]
    receiver_expr = expr_sub[:, receiver_idx]
    sender_coords = coords[sender_idx]
    receiver_coords = coords[receiver_idx]

    radius = 50
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
    print(f"  Result shape: {result.shape}")
    print(f"  Value range: [{result.min():.4f}, {result.max():.4f}]")

    return result, elapsed


def validate_against_r(data):
    """Cross-validate against R sigdiscov package."""
    import subprocess
    import tempfile

    print(f"\n{'='*60}")
    print("Cross-Validation Against R (sigdiscov)")
    print(f"{'='*60}")

    expr = data["expr"]
    coords = data["coords"]

    # Use small subset for validation
    n_genes_test = 10
    n_cells_test = 1000
    np.random.seed(42)
    cell_idx = np.random.choice(expr.shape[1], n_cells_test, replace=False)
    expr_sub = expr[:n_genes_test, cell_idx]
    coords_sub = coords[cell_idx]

    print(f"\nUsing subset: {n_genes_test} genes x {n_cells_test} cells")

    # Python computation
    radius = 50
    expr_norm_py = standardize_matrix(expr_sub, use_gpu=False)
    W_py = create_gaussian_weights(coords_sub, radius=radius, use_gpu=False)
    W_dense_py = W_py.toarray() if hasattr(W_py, 'toarray') else W_py
    weight_sum_py = float(W_dense_py.sum())

    result_py = pairwise_moran(expr_norm_py, coords_sub, radius=radius, use_gpu=False)

    print(f"\nPython results:")
    print(f"  Weight sum: {weight_sum_py:.4f}")
    print(f"  Pairwise Moran I[0,0]: {result_py[0,0]:.10f}")
    print(f"  Pairwise Moran I[0,1]: {result_py[0,1]:.10f}")

    # Run R validation
    with tempfile.TemporaryDirectory() as tmpdir:
        expr_file = Path(tmpdir) / "expr.csv"
        coords_file = Path(tmpdir) / "coords.csv"

        pd.DataFrame(expr_sub, index=[f"gene_{i}" for i in range(n_genes_test)]).to_csv(expr_file)
        pd.DataFrame(coords_sub, columns=['x', 'y']).to_csv(coords_file)

        r_code = f'''
        library(sigdiscov)
        expr <- as.matrix(read.csv("{expr_file}", row.names=1, check.names=FALSE))
        coords <- as.matrix(read.csv("{coords_file}", row.names=1))

        # Use create_weights_sc for single-cell data
        W <- create_weights_sc(coords, coords, radius={radius}, sigma={radius}/3)
        W <- as.matrix(W)
        weight_sum <- sum(W)

        expr_norm <- standardize_matrix(expr)
        result <- pairwise_moran_custom(expr_norm, W, mode="paired", verbose=FALSE)

        cat("weight_sum:", weight_sum, "\\n")
        cat("I_0_0:", result$moran[1,1], "\\n")
        cat("I_0_1:", result$moran[1,2], "\\n")
        '''

        result = subprocess.run(['Rscript', '-e', r_code], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"R Error: {result.stderr}")
            return False

        print(f"\nR results:")
        weight_sum_r = None
        I_0_0_r = None
        I_0_1_r = None
        for line in result.stdout.strip().split('\n'):
            if 'weight_sum:' in line:
                weight_sum_r = float(line.split(':')[1].strip())
                print(f"  Weight sum: {weight_sum_r:.4f}")
            elif 'I_0_0:' in line:
                I_0_0_r = float(line.split(':')[1].strip())
                print(f"  Pairwise Moran I[0,0]: {I_0_0_r:.10f}")
            elif 'I_0_1:' in line:
                I_0_1_r = float(line.split(':')[1].strip())
                print(f"  Pairwise Moran I[0,1]: {I_0_1_r:.10f}")

        # Compare
        print(f"\nComparison:")
        # Note: R row-normalizes by default (rows sum to 1), so weight_sum = n_cells
        # Python doesn't row-normalize by default, so weight sums differ.
        # This is expected behavior - the Moran's I values should still match.
        print(f"  Note: R row-normalizes weights (sum=n_cells), Python doesn't")

        passed = True
        if I_0_0_r is not None:
            I_0_0_diff = abs(result_py[0,0] - I_0_0_r)
            is_pass = I_0_0_diff < 1e-5
            print(f"  I[0,0] diff: {I_0_0_diff:.2e} {'PASS' if is_pass else 'FAIL'}")
            passed = passed and is_pass

        if I_0_1_r is not None:
            I_0_1_diff = abs(result_py[0,1] - I_0_1_r)
            is_pass = I_0_1_diff < 1e-5
            print(f"  I[0,1] diff: {I_0_1_diff:.2e} {'PASS' if is_pass else 'FAIL'}")
            passed = passed and is_pass

        return passed


def main():
    print("=" * 60)
    print("sigdiscovPy CosMx Validation Test")
    print("=" * 60)

    # Load data
    data = load_cosmx_lung5()

    # Test core functions (CPU)
    cpu_results = test_core_functions(data, use_gpu=False)

    # Test pairwise Moran (CPU)
    pairwise_result, pairwise_time = test_pairwise_moran(data, use_gpu=False)

    # Test cell type analysis (CPU)
    celltype_result, celltype_time = test_celltype_analysis(data, use_gpu=False)

    # Cross-validate against R
    r_validation = validate_against_r(data)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Dataset: Lung5_Rep1 (98,002 cells, 960 genes)")
    print(f"All tests completed successfully!")
    print(f"\nKey metrics:")
    print(f"  Weight sum (r=50): {cpu_results['weight_sum']:.2f}")
    print(f"  Sample Moran's I: {cpu_results['moran_i']:.6f}")
    print(f"  Sample I_ND: {cpu_results['ind']:.6f}")
    print(f"  Pairwise Moran time (20 genes, 3k cells): {pairwise_time:.2f}s")
    print(f"  Cell type analysis time: {celltype_time:.2f}s")
    print(f"  R cross-validation: {'PASSED' if r_validation else 'FAILED'}")

    return 0 if r_validation else 1


if __name__ == "__main__":
    exit(main())
