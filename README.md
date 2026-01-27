# sigdiscovPy

GPU-accelerated spatial signature discovery for spatial transcriptomics.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/psychemistz/sigdiscovPy/actions/workflows/test.yml/badge.svg)](https://github.com/psychemistz/sigdiscovPy/actions/workflows/test.yml)

Python implementation of spatial correlation algorithms with GPU acceleration (CuPy) and automatic CPU fallback. Produces **identical outputs** to the [sigdiscov R package](https://github.com/psychemistz/sigdiscov).

## Installation

```bash
pip install sigdiscovpy          # CPU-only
pip install sigdiscovpy[gpu]     # With GPU support
```

## Quick Start

```python
import numpy as np
from sigdiscovpy import pairwise_moran

# Expression data (genes x cells) and coordinates
expr = np.random.randn(100, 1000).astype(np.float32)
coords = np.random.randn(1000, 2).astype(np.float32) * 100

# Compute pairwise Moran's I matrix
I_matrix = pairwise_moran(expr, coords, radius=50)
```

### Permutation Testing

```python
from sigdiscovpy.stats import permutation_test, apply_fdr_correction

result = permutation_test(z_f, lag_g, n_permutations=999)
adjusted_pvalues = apply_fdr_correction(pvalues, method='bh')
```

## Simulation Framework

```python
from sigdiscovpy.simulation import SimulationPresets, UnifiedSimulation

config = SimulationPresets.small_scale()
sim = UnifiedSimulation(config)
result = sim.run_single(receiver_fraction=0.2)
```

**Presets**: `default()`, `small_scale()`, `large_scale()`, `strong_signal()`, `weak_signal()`, `high_noise()`, `low_noise()`, `long_range()`, `short_range()`

## Key Functions

| Function | Description |
|----------|-------------|
| `pairwise_moran(expr, coords, radius)` | Pairwise Moran's I matrix |
| `compute_ind_from_lag(z_f, lag_g)` | I_ND (cosine similarity) |
| `create_gaussian_weights(coords, radius)` | Gaussian distance decay weights |
| `permutation_test(z_f, lag_g)` | Permutation significance test |
| `apply_fdr_correction(pvalues, method)` | FDR correction (bh, by, bonferroni) |

## Mathematical Background

| Metric | Formula |
|--------|---------|
| Bivariate Moran's I | `I = (z_f' * W * z_g) / weight_sum` |
| I_ND (cosine similarity) | `I_ND = (z_f' * lag_g) / (‖z_f‖ * ‖lag_g‖)` |
| Gaussian weights | `w(d) = exp(-(d/σ)² / 2), σ = radius/3` |
| Diffusion model | `c(r) = (A / 4πD) * K₀(r/λ), λ = √(D/k)` |

## Performance Benchmarks

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Pairwise Moran (1k genes, 5k cells) | ~10s | ~0.1s | 100x |
| Permutation Test (999 perms) | ~60s | ~0.5s | 120x |
| Weight Matrix (50k cells) | ~30s | ~1s | 30x |

## Testing

```bash
pytest tests/ -v
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

MIT License | [sigdiscov R package](https://github.com/psychemistz/sigdiscov)
