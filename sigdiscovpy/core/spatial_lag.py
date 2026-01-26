"""
Spatial lag computation.

The spatial lag is the weighted average of neighboring values:
    lag_i = sum_j(w_ij * z_j)

This is the core operation for spatial correlation analysis.
"""

from typing import Union
import numpy as np
from scipy import sparse as sp_sparse
from sigdiscovpy.gpu.backend import get_array_module, GPU_AVAILABLE, ensure_numpy


def compute_spatial_lag(
    W,
    z,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Compute spatial lag of expression vector.

    Formula: lag = W @ z

    Parameters
    ----------
    W : sparse matrix or np.ndarray
        Spatial weight matrix (n x m), typically row-normalized.
    z : array-like
        Expression values (length m).
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    np.ndarray
        Spatial lag vector (length n).

    Examples
    --------
    >>> from scipy.sparse import random as sp_random
    >>> W = sp_random(100, 100, density=0.1, format='csr')
    >>> z = np.random.randn(100)
    >>> lag = compute_spatial_lag(W, z)
    >>> lag.shape
    (100,)

    Notes
    -----
    The spatial lag represents the weighted average of neighboring values.
    For row-normalized W, this is the expected value of z in the neighborhood.
    """
    z = np.asarray(z, dtype=np.float64).ravel()

    if use_gpu and GPU_AVAILABLE:
        return _compute_spatial_lag_gpu(W, z)

    # CPU implementation
    if sp_sparse.issparse(W):
        lag = np.asarray(W @ z).ravel()
    else:
        lag = np.dot(W, z)

    return lag


def _compute_spatial_lag_gpu(W, z):
    """GPU implementation of spatial lag."""
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    z_gpu = cp.asarray(z, dtype=cp.float64)

    if sp_sparse.issparse(W):
        # Convert to CuPy sparse
        W_gpu = cp_sparse.csr_matrix(W.astype(np.float64))
        lag_gpu = W_gpu @ z_gpu
    else:
        W_gpu = cp.asarray(W, dtype=cp.float64)
        lag_gpu = W_gpu @ z_gpu

    return ensure_numpy(lag_gpu)


def compute_spatial_lag_batch(
    W,
    Z,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Compute spatial lags for all genes at once.

    Formula: lag_G = W @ Z

    Parameters
    ----------
    W : sparse matrix or np.ndarray
        Spatial weight matrix (n x m), row-normalized.
    Z : array-like
        Expression matrix (m x n_genes), genes in columns.
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    np.ndarray
        Spatial lag matrix (n x n_genes).

    Examples
    --------
    >>> from scipy.sparse import random as sp_random
    >>> n, n_genes = 100, 50
    >>> W = sp_random(n, n, density=0.1, format='csr')
    >>> W = W / W.sum(axis=1)  # Row normalize
    >>> Z = np.random.randn(n, n_genes)
    >>> lag_G = compute_spatial_lag_batch(W, Z)
    >>> lag_G.shape
    (100, 50)

    Notes
    -----
    This is the workhorse for genome-wide analysis. Compute W @ Z once,
    then use compute_metric_batch() for all factor-gene pairs.
    """
    Z = np.asarray(Z, dtype=np.float64)

    if use_gpu and GPU_AVAILABLE:
        return _compute_spatial_lag_batch_gpu(W, Z)

    # CPU implementation
    if sp_sparse.issparse(W):
        lag_G = np.asarray(W @ Z)
    else:
        lag_G = np.dot(W, Z)

    return lag_G


def _compute_spatial_lag_batch_gpu(W, Z):
    """GPU implementation of batch spatial lag."""
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    Z_gpu = cp.asarray(Z, dtype=cp.float64)

    if sp_sparse.issparse(W):
        W_gpu = cp_sparse.csr_matrix(W.astype(np.float64))
        lag_G_gpu = W_gpu @ Z_gpu
    else:
        W_gpu = cp.asarray(W, dtype=cp.float64)
        lag_G_gpu = W_gpu @ Z_gpu

    return ensure_numpy(lag_G_gpu)


def compute_spatial_lag_chunked(
    W,
    Z,
    chunk_size: int = 1000,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Compute spatial lags in chunks to manage memory.

    Parameters
    ----------
    W : sparse matrix
        Spatial weight matrix (n x n).
    Z : array-like
        Expression matrix (n x n_genes).
    chunk_size : int, default=1000
        Number of genes per chunk.
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    np.ndarray
        Spatial lag matrix (n x n_genes).
    """
    Z = np.asarray(Z, dtype=np.float64)
    n, n_genes = Z.shape

    result = np.zeros((n, n_genes), dtype=np.float64)

    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        Z_chunk = Z[:, start:end]
        result[:, start:end] = compute_spatial_lag_batch(W, Z_chunk, use_gpu=use_gpu)

    return result
