# sigdiscovPy

GPU-accelerated spatial signature discovery for spatial transcriptomics.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**sigdiscovPy** is a Python package for spatial signature discovery in spatial transcriptomics data. It provides GPU-accelerated implementations of spatial correlation algorithms with optional CPU fallback.

This package is designed to produce **identical outputs** to the [sigdiscov R package](https://github.com/psychemistz/sigdiscov) (>0.99 correlation).

### Key Features

- **Bivariate Moran's I** - Spatial autocorrelation metric
- **I_ND (Cosine Similarity)** - Normalized directional Moran's I
- **Pairwise Moran's I Matrix** - Gene×gene spatial correlation matrix
- **GPU Acceleration** - 50-500x speedup with CuPy
- **Permutation Testing** - GPU-parallel significance testing
- **FDR Correction** - Benjamini-Hochberg, Benjamini-Yekutieli, Bonferroni

## Installation

### CPU-only (NumPy backend)

```bash
pip install sigdiscovpy
```

### With GPU support (CuPy backend)

```bash
pip install sigdiscovpy[gpu]
```

### Development installation

```bash
git clone https://github.com/psychemistz/sigdiscovPy.git
cd sigdiscovPy
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import numpy as np
from sigdiscovpy import pairwise_moran, create_gaussian_weights

# Load expression data (genes x cells)
expr = np.random.randn(100, 1000)
coords = np.random.randn(1000, 2) * 100

# Compute pairwise Moran's I matrix
I_matrix = pairwise_moran(expr, coords, radius=50)
print(f"Shape: {I_matrix.shape}")  # (100, 100)
```

### With AnnData

```python
from sigdiscovpy.io import load_anndata
from sigdiscovpy import pairwise_moran

# Load spatial data
data = load_anndata("spatial_data.h5ad")

# Compute pairwise Moran's I
I_matrix = pairwise_moran(data['expr'], data['coords'], radius=100)
```

### GPU Acceleration

```python
from sigdiscovpy import pairwise_moran, GPU_AVAILABLE

print(f"GPU available: {GPU_AVAILABLE}")

# GPU is used by default when available
I_matrix = pairwise_moran(expr, coords, radius=50, use_gpu=True)

# Force CPU computation
I_matrix_cpu = pairwise_moran(expr, coords, radius=50, use_gpu=False)
```

### Permutation Testing

```python
from sigdiscovpy.stats import batch_permutation_test
from sigdiscovpy.core import compute_spatial_lag_batch, standardize_matrix

# Standardize expression
expr_z = standardize_matrix(expr)

# Compute spatial lags
lag_G = compute_spatial_lag_batch(W, expr_z.T)

# Run permutation test for one factor against all genes
observed, pvalues = batch_permutation_test(
    expr_z[0],  # Factor expression
    lag_G,       # Spatial lags
    n_permutations=999,
)
```

## Core Functions

### Metrics

| Function | Description |
|----------|-------------|
| `compute_moran_from_lag(z_f, lag_g)` | Bivariate Moran's I from spatial lag |
| `compute_ind_from_lag(z_f, lag_g)` | I_ND (cosine similarity) from spatial lag |
| `compute_metric_batch(z_f, lag_G)` | Batch metrics for all genes |

### Weights

| Function | Description |
|----------|-------------|
| `create_gaussian_weights(coords, radius)` | Gaussian distance decay weights |
| `create_ring_weights(coords, outer_r, inner_r)` | Annular weights |
| `row_normalize_weights(W)` | Row-normalize weight matrix |

### Analysis

| Function | Description |
|----------|-------------|
| `pairwise_moran(expr, coords, radius)` | Pairwise Moran's I matrix |
| `pairwise_moran_directional(...)` | Directional pairwise Moran |

### Statistics

| Function | Description |
|----------|-------------|
| `permutation_test(z_f, lag_g)` | Single permutation test |
| `batch_permutation_test(z_f, lag_G)` | Batch permutation test |
| `apply_fdr_correction(pvalues)` | FDR correction |

## Mathematical Background

### Bivariate Moran's I

```
I = (z_f' × W × z_g) / weight_sum
```

### I_ND (Cosine Similarity)

```
I_ND = (z_f' × lag_g) / (||z_f|| × ||lag_g||)
```

where `lag_g = W × z_g` is the spatial lag.

### Gaussian Weights

```
w(d) = exp(-(d/σ)² / 2), σ = radius/3
```

## Performance

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Pairwise Moran (1000 genes) | 10s | 0.1s | 100x |
| Permutation Test (999 perms) | 60s | 0.5s | 120x |
| Weight Matrix (50k cells) | 30s | 1s | 30x |

## Comparison with R Package

sigdiscovPy is designed to produce identical outputs to the sigdiscov R package:

| Metric | Correlation | Max Difference |
|--------|-------------|----------------|
| Moran's I | >0.9999 | <1e-10 |
| I_ND | >0.9999 | <1e-10 |
| Weight Sum | Exact | 0 |

## Citation

```bibtex
@article{ru2025secact,
  title={Inference of secreted protein activities in intercellular communication},
  author={Ru, Beibei and Gong, Lanqi and Yang, Emily and Park, Seongyong and Zaki, George and Aldape, Kenneth and Wakefield, Lalage and Jiang, Peng},
  journal={SecAct},
  year={2025}
}
```

## License

MIT License

## Related Projects

- [sigdiscov](https://github.com/psychemistz/sigdiscov) - R package (CPU-optimized with RcppArmadillo)
