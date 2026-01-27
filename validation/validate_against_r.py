#!/usr/bin/env python3
"""
Cross-validation script: sigdiscovPy vs sigdiscov R package.

This script validates Python implementations against the R reference
using real Visium data.

Usage:
    python validation/validate_against_r.py

Requirements:
    - sigdiscovPy installed
    - R with sigdiscov package installed
    - Test data: /Users/seongyongpark/project/sigdiscov/dataset/visium/3_vst.tsv
"""

import subprocess
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sigdiscovpy.core.normalization import standardize_matrix
from sigdiscovpy.core.weights import create_gaussian_weights
from sigdiscovpy.core.spatial_lag import compute_spatial_lag
from sigdiscovpy.analysis.pairwise_moran import pairwise_moran


def parse_spot_names(spot_names):
    """Parse spot names in 'ROWxCOL' format to coordinates."""
    coords = []
    for name in spot_names:
        parts = name.split('x')
        row, col = int(parts[0]), int(parts[1])
        coords.append([row, col])
    return np.array(coords)


def convert_to_spatial_coords(array_coords, scale=100.0):
    """Convert array coordinates to spatial coordinates with hex shift."""
    hex_shift = 0.5 * np.sqrt(3)
    spatial = np.zeros_like(array_coords, dtype=np.float64)
    spatial[:, 0] = array_coords[:, 0] * hex_shift * scale
    spatial[:, 1] = array_coords[:, 1] * scale
    return spatial


def load_visium_data(filepath):
    """Load Visium VST data."""
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    df.columns = [col.replace('X', '') for col in df.columns]
    expr = df.values.astype(np.float64)
    gene_names = df.index.tolist()
    spot_names = df.columns.tolist()
    return expr, gene_names, spot_names


def run_r_validation(expr_file, coords_file, output_file):
    """Run R validation script and return results."""
    r_code = f'''
    library(sigdiscov)

    # Load data
    expr <- as.matrix(read.csv("{expr_file}", row.names=1, check.names=FALSE))
    coords <- as.matrix(read.csv("{coords_file}", row.names=1))

    # Create weight matrix
    W <- cpp_create_weight_matrix(coords, 300)
    weight_sum <- sum(W)

    # Normalize expression
    expr_norm <- standardize_matrix(expr)

    # Compute pairwise Moran's I (first 10 genes)
    result <- cpp_pairwise_moran(expr_norm[1:10,], W)

    # Save results
    output <- list(
        weight_sum = weight_sum,
        pairwise_moran = result
    )
    saveRDS(output, "{output_file}")

    cat("weight_sum:", weight_sum, "\\n")
    cat("pairwise_moran[1,1]:", result[1,1], "\\n")
    '''

    result = subprocess.run(
        ['Rscript', '-e', r_code],
        capture_output=True, text=True
    )
    return result.stdout, result.stderr


def main():
    print("=" * 60)
    print("sigdiscovPy vs sigdiscov R Validation")
    print("=" * 60)

    # Load test data
    data_path = Path("/Users/seongyongpark/project/sigdiscov/dataset/visium/3_vst.tsv")
    if not data_path.exists():
        print(f"ERROR: Test data not found at {data_path}")
        return 1

    print(f"\nLoading data from: {data_path}")
    expr, gene_names, spot_names = load_visium_data(data_path)
    print(f"  Expression matrix: {expr.shape[0]} genes x {expr.shape[1]} spots")

    # Parse coordinates
    array_coords = parse_spot_names(spot_names)
    coords = convert_to_spatial_coords(array_coords)
    print(f"  Coordinates: {coords.shape}")

    # Python computations
    print("\n--- Python (sigdiscovPy) ---")

    # Weight matrix
    W_py = create_gaussian_weights(coords, radius=300, use_gpu=False)
    W_dense = W_py.toarray() if hasattr(W_py, 'toarray') else W_py
    weight_sum_py = float(W_dense.sum())
    print(f"Weight sum: {weight_sum_py:.6f}")

    # Normalize
    expr_norm_py = standardize_matrix(expr, use_gpu=False)

    # Pairwise Moran (first 10 genes)
    result_py = pairwise_moran(expr_norm_py[:10, :], coords, radius=300, use_gpu=False)
    print(f"Pairwise Moran I[0,0]: {result_py[0, 0]:.10f}")

    # R computations
    print("\n--- R (sigdiscov) ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        expr_file = Path(tmpdir) / "expr.csv"
        coords_file = Path(tmpdir) / "coords.csv"
        output_file = Path(tmpdir) / "output.rds"

        # Save data for R
        pd.DataFrame(expr, index=gene_names).to_csv(expr_file)
        pd.DataFrame(coords, columns=['x', 'y']).to_csv(coords_file)

        # Run R
        stdout, stderr = run_r_validation(expr_file, coords_file, output_file)

        if stderr and "Error" in stderr:
            print(f"R Error: {stderr}")
            return 1

        # Parse R output
        for line in stdout.strip().split('\n'):
            if 'weight_sum:' in line:
                weight_sum_r = float(line.split(':')[1].strip())
                print(f"Weight sum: {weight_sum_r:.6f}")
            elif 'pairwise_moran[1,1]:' in line:
                moran_r = float(line.split(':')[1].strip())
                print(f"Pairwise Moran I[0,0]: {moran_r:.10f}")

    # Comparison
    print("\n--- Validation Results ---")

    weight_diff = abs(weight_sum_py - weight_sum_r)
    print(f"Weight sum difference: {weight_diff:.6e} {'PASS' if weight_diff < 0.01 else 'FAIL'}")

    moran_diff = abs(result_py[0, 0] - moran_r)
    print(f"Moran I[0,0] difference: {moran_diff:.6e} {'PASS' if moran_diff < 1e-6 else 'FAIL'}")

    # Overall result
    if weight_diff < 0.01 and moran_diff < 1e-6:
        print("\n" + "=" * 60)
        print("VALIDATION PASSED: Python matches R implementation")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("VALIDATION FAILED: Results differ beyond tolerance")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
