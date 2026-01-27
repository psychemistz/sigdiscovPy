# sigdiscovPy

GPU-accelerated spatial signature discovery for spatial transcriptomics.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/psychemistz/sigdiscovPy/actions/workflows/test.yml/badge.svg)](https://github.com/psychemistz/sigdiscovPy/actions/workflows/test.yml)
[![Lint](https://github.com/psychemistz/sigdiscovPy/actions/workflows/lint.yml/badge.svg)](https://github.com/psychemistz/sigdiscovPy/actions/workflows/lint.yml)

## Overview

**sigdiscovPy** is a Python package for spatial signature discovery in spatial transcriptomics data. It provides GPU-accelerated implementations of spatial correlation algorithms with automatic CPU fallback.

This package is designed to produce **identical outputs** to the [sigdiscov R package](https://github.com/psychemistz/sigdiscov) (>0.99 correlation).

### Key Features

- **Bivariate Moran's I** - Spatial autocorrelation metric
- **I_ND (Cosine Similarity)** - Normalized directional Moran's I
- **Pairwise Moran's I Matrix** - Gene-by-gene spatial correlation matrix
- **GPU Acceleration** - 50-500x speedup with CuPy backend
- **Permutation Testing** - GPU-parallel significance testing
- **FDR Correction** - Benjamini-Hochberg, Benjamini-Yekutieli, Bonferroni
- **Simulation Framework** - Generate synthetic spatial data with known ground truth

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
expr = np.random.randn(100, 1000).astype(np.float32)
coords = np.random.randn(1000, 2).astype(np.float32) * 100

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
from sigdiscovpy.stats import permutation_test, batch_permutation_test
from sigdiscovpy.core import compute_spatial_lag, standardize_matrix, create_gaussian_weights

# Create weight matrix and standardize expression
W = create_gaussian_weights(coords, radius=100)
expr_z = standardize_matrix(expr)

# Single permutation test
z_f = expr_z[0]  # Factor expression
lag_g = compute_spatial_lag(W, expr_z[1])  # Spatial lag of target

result = permutation_test(z_f, lag_g, n_permutations=999)
print(f"I_ND: {result['observed']:.4f}, p-value: {result['pvalue']:.4f}")

# Batch permutation test for one factor against all genes
observed, pvalues = batch_permutation_test(z_f, lag_G, n_permutations=999)
```

### FDR Correction

```python
from sigdiscovpy.stats import apply_fdr_correction

# Apply Benjamini-Hochberg FDR correction
adjusted_pvalues = apply_fdr_correction(pvalues, method='bh')

# Count significant genes at FDR < 0.05
n_significant = np.sum(adjusted_pvalues < 0.05)
```

## Simulation Framework

Generate synthetic spatial transcriptomics data with known ground truth for method validation.

```python
from sigdiscovpy.simulation import SimulationPresets, UnifiedSimulation

# Use a preset configuration
config = SimulationPresets.small_scale()
config.analysis.radii = [100, 200, 300, 400, 500]

# Run simulation
sim = UnifiedSimulation(config)
result = sim.run_single(receiver_fraction=0.2)

# Access results
print(f"Lambda (diffusion length): {result['lambda']:.0f} um")
print(f"I_ND curve:")
for point in result['ind_curve']:
    print(f"  r={point['radius']}: I_ND={point['I_ND']:.4f}")
```

### Available Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `default()` | Balanced configuration (100k cells) | General analysis |
| `small_scale()` | Smaller dataset (5k cells) | Quick testing |
| `large_scale()` | Large dataset (500k cells) | Production runs |
| `strong_signal()` | High fold change, clear signal | Method validation |
| `weak_signal()` | Low fold change, noisy signal | Sensitivity testing |
| `high_noise()` | High expression variance | Robustness testing |
| `low_noise()` | Low expression variance | Baseline comparison |
| `long_range()` | Large diffusion length | Long-range signaling |
| `short_range()` | Small diffusion length | Contact-dependent signaling |

### Custom Configuration

```python
from sigdiscovpy.simulation import SimulationConfig, DomainConfig, CellTypeConfig

config = SimulationConfig(
    domain=DomainConfig(n_cells=10000, max_radius=2000),
    cell_types=CellTypeConfig(n_active_senders=50, receiver_fractions=[0.1, 0.2, 0.3]),
)
```

## Package Structure

```
sigdiscovpy/
├── core/                    # Core algorithms
│   ├── metrics.py          # Moran's I, I_ND computation
│   ├── weights.py          # Weight matrix creation
│   ├── spatial_lag.py      # Spatial lag computation
│   └── normalization.py    # Data standardization
├── analysis/               # High-level analysis functions
│   ├── pairwise_moran.py   # Pairwise Moran's I matrix
│   ├── genomewide.py       # Genome-wide analysis
│   ├── celltype_pair.py    # Cell type pair analysis
│   └── delta_i.py          # Delta I computation
├── stats/                  # Statistical testing
│   ├── permutation.py      # Permutation tests
│   └── fdr.py              # FDR correction
├── simulation/             # Simulation framework
│   ├── config/             # Configuration classes
│   ├── domain/             # Spatial domain generation
│   ├── physics/            # Diffusion modeling
│   ├── expression/         # Expression generation
│   └── runner.py           # Simulation orchestrator
├── io/                     # Data I/O
│   ├── loaders.py          # AnnData, CosMx loaders
│   └── hdf5.py             # HDF5 I/O
├── neighbors/              # Neighbor search
│   └── kdtree.py           # KD-tree implementation
└── gpu/                    # GPU backend
    └── backend.py          # CuPy/NumPy switching
```

## API Reference

### Core Metrics

| Function | Description |
|----------|-------------|
| `compute_moran_from_lag(z_f, lag_g)` | Bivariate Moran's I from spatial lag |
| `compute_ind_from_lag(z_f, lag_g)` | I_ND (cosine similarity) from spatial lag |
| `compute_metric_batch(z_f, lag_G)` | Batch metrics for all genes |

### Weight Matrices

| Function | Description |
|----------|-------------|
| `create_gaussian_weights(coords, radius)` | Gaussian distance decay weights |
| `create_ring_weights(coords, outer_r, inner_r)` | Annular/ring weights |
| `row_normalize_weights(W)` | Row-normalize weight matrix |

### Analysis Functions

| Function | Description |
|----------|-------------|
| `pairwise_moran(expr, coords, radius)` | Pairwise Moran's I matrix |
| `pairwise_moran_directional(...)` | Directional sender-receiver analysis |

### Statistical Testing

| Function | Description |
|----------|-------------|
| `permutation_test(z_f, lag_g)` | Single permutation test |
| `batch_permutation_test(z_f, lag_G)` | Batch permutation test |
| `apply_fdr_correction(pvalues, method)` | FDR correction (bh, by, bonferroni) |

### Simulation

| Class/Function | Description |
|----------------|-------------|
| `SimulationPresets` | Factory for preset configurations |
| `UnifiedSimulation` | Main simulation orchestrator |
| `SimulationConfig` | Complete simulation configuration |

## Mathematical Background

### Bivariate Moran's I

```
I = (z_f' * W * z_g) / weight_sum
```

### I_ND (Normalized Directional Moran's I)

```
I_ND = (z_f' * lag_g) / (||z_f|| * ||lag_g||)
```

where `lag_g = W * z_g` is the spatial lag.

### Gaussian Weights

```
w(d) = exp(-(d/sigma)^2 / 2), sigma = radius/3
```

### Diffusion Model (Simulation)

The simulation uses a Green's function solution for steady-state 2D diffusion:

```
c(r) = (A / 4*pi*D) * K_0(r/lambda)
```

where `lambda = sqrt(D/k)` is the characteristic diffusion length.

## Performance Benchmarks

| Operation | CPU (NumPy) | GPU (CuPy) | Speedup |
|-----------|-------------|------------|---------|
| Pairwise Moran (1000 genes, 5000 cells) | ~10s | ~0.1s | 100x |
| Permutation Test (999 permutations) | ~60s | ~0.5s | 120x |
| Weight Matrix (50k cells) | ~30s | ~1s | 30x |
| Standardization (10k x 50k) | ~5s | ~0.1s | 50x |

Run benchmarks:
```bash
python benchmarks/benchmark_gpu_vs_cpu.py
```

## Comparison with R Package

sigdiscovPy produces identical outputs to the sigdiscov R package:

| Metric | Correlation | Max Difference |
|--------|-------------|----------------|
| Moran's I | >0.9999 | <1e-10 |
| I_ND | >0.9999 | <1e-10 |
| Weight Sum | Exact | 0 |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sigdiscovpy --cov-report=html

# Run GPU tests only (requires CuPy)
pytest tests/test_gpu.py -v

# Run simulation tests
pytest tests/test_simulation.py -v
```

## Development

### Code Style

```bash
# Format code
black sigdiscovpy/ tests/

# Lint
ruff check sigdiscovpy/ tests/

# Type check
mypy sigdiscovpy/
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

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
