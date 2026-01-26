# sigdiscovPy - GPU-Accelerated Spatial Signature Discovery

## Project Overview

**sigdiscovPy** is the Python companion to the sigdiscov R package, providing GPU-accelerated implementations of spatial correlation algorithms for spatial transcriptomics analysis.

**Location**: `/Users/seongyongpark/project/sigdiscovPy`
**Language**: Python with CuPy GPU backend
**License**: MIT

## Package Structure

```
sigdiscovPy/
├── sigdiscovpy/
│   ├── __init__.py              # Main exports
│   ├── core/
│   │   ├── metrics.py           # Moran's I, I_ND (cosine similarity)
│   │   ├── weights.py           # Weight matrix creation (Gaussian, ring)
│   │   ├── spatial_lag.py       # W * z computation
│   │   └── normalization.py     # Global z-score standardization
│   ├── analysis/
│   │   └── pairwise_moran.py    # Pairwise Moran's I matrix
│   ├── stats/
│   │   ├── permutation.py       # Permutation testing (GPU-parallel)
│   │   └── fdr.py               # FDR correction (BH, BY, Bonferroni)
│   ├── io/
│   │   ├── loaders.py           # AnnData, CosMx loaders
│   │   └── hdf5.py              # HDF5 I/O
│   ├── neighbors/
│   │   └── kdtree.py            # KD-tree neighbor search
│   └── gpu/
│       └── backend.py           # CuPy/NumPy backend switching
├── tests/
│   ├── test_metrics.py
│   ├── test_normalization.py
│   ├── test_weights.py
│   ├── test_pairwise_moran.py
│   ├── test_fdr.py
│   └── test_cross_validation.py  # R vs Python validation
├── .claude/agents/               # Claude Code agent configurations
├── pyproject.toml
└── README.md
```

## Key Formulas

### Bivariate Moran's I
```
I = (z_f.T @ W @ z_g) / weight_sum
```

### I_ND (Cosine Similarity)
```
I_ND = (z_f.T @ lag_g) / (||z_f|| * ||lag_g||)
```
where `lag_g = W @ z_g`

### Gaussian Weights
```
w(d) = exp(-(d/σ)² / 2), σ = radius/3
```

## CRITICAL: Global Normalization

**All spatial analyses MUST use GLOBAL normalization** - computing mean/std across ALL cells.

```python
# Correct (matching R package and Python v7):
X_mean = X.mean(axis=0, keepdims=True)  # Mean across ALL cells
X_std = X.std(axis=0, keepdims=True)    # Std across ALL cells
X_normalized = (X - X_mean) / (X_std + 1e-10)

# Then extract subsets from globally normalized data
sender_expr = X_normalized[sender_indices]
```

## GPU Backend

```python
from sigdiscovpy.gpu.backend import get_array_module, GPU_AVAILABLE

xp = get_array_module(use_gpu=True)  # Returns cupy or numpy
x = xp.array([1, 2, 3])
```

## Build Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sigdiscovpy

# Type checking
mypy sigdiscovpy/
```

## R Package Reference

- R package: `/Users/seongyongpark/project/sigdiscov/`
- Python reference: `/Users/seongyongpark/project/sigdiscov/development/python_reference/genomewide_interaction_v7.py`

## Validation Requirements

- Numerical accuracy: >0.99 correlation with R outputs
- Tolerance: rtol=1e-10 for floating point comparison
- All radii should match R implementation
