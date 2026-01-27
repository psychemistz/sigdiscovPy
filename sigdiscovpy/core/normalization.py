"""
Normalization functions for expression data.

Global normalization is CRITICAL for spatial correlation analysis.
All functions compute mean/std across ALL cells, not subsets.
"""

import numpy as np
from sigdiscovpy.gpu.backend import get_array_module, ensure_numpy


def standardize_vector(x, use_gpu: bool = True) -> np.ndarray:
    """
    Z-score standardization of a vector.

    Computes (x - mean(x)) / std(x) using population standard deviation.

    Parameters
    ----------
    x : array-like
        Input vector of length n.
    use_gpu : bool, default=True
        Whether to use GPU acceleration if available.

    Returns
    -------
    np.ndarray
        Standardized vector with mean=0, std=1.
        Returns zeros if std < 1e-10 (constant vector).

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> z = standardize_vector(x)
    >>> np.abs(z.mean()) < 1e-10
    True
    >>> np.abs(z.std() - 1.0) < 1e-10
    True

    Notes
    -----
    Uses population standard deviation (N, not N-1) to match R implementation.
    """
    xp = get_array_module(use_gpu)
    x_arr = xp.asarray(x, dtype=xp.float64)

    n = x_arr.shape[0]
    mean_x = xp.mean(x_arr)
    # Population std: sqrt(sum((x - mean)^2) / N)
    var_x = xp.sum((x_arr - mean_x) ** 2) / n
    std_x = xp.sqrt(var_x)

    if float(std_x) < 1e-10:
        return ensure_numpy(xp.zeros(n, dtype=xp.float64))

    result = (x_arr - mean_x) / std_x
    return ensure_numpy(result)


def standardize_matrix(
    X,
    axis: int = 1,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Gene-wise (row-wise) z-score standardization.

    Each gene is standardized to have mean=0 and std=1 across all cells/spots.
    This is GLOBAL normalization - uses all cells, not subsets.

    Parameters
    ----------
    X : array-like
        Expression matrix of shape (n_genes, n_cells) or (n_cells, n_genes).
    axis : int, default=1
        Axis along which to standardize:
        - axis=1: Standardize each row (genes x cells format, default)
        - axis=0: Standardize each column (cells x genes format)
    use_gpu : bool, default=True
        Whether to use GPU acceleration if available.

    Returns
    -------
    np.ndarray
        Standardized matrix with same shape as input.
        Rows/columns with zero variance are set to zeros.

    Examples
    --------
    >>> X = np.random.randn(100, 1000)  # 100 genes x 1000 cells
    >>> Z = standardize_matrix(X, axis=1)
    >>> np.allclose(Z.mean(axis=1), 0, atol=1e-10)
    True
    >>> np.allclose(Z.std(axis=1), 1, atol=1e-10)
    True

    Notes
    -----
    CRITICAL: This function implements GLOBAL normalization matching the
    R sigdiscov package and Python v7 reference implementation.

    Uses population standard deviation (N, not N-1).
    """
    xp = get_array_module(use_gpu)
    X_arr = xp.asarray(X, dtype=xp.float64)

    n_rows, n_cols = X_arr.shape

    if axis == 1:
        # Standardize each row (genes x cells format)
        # Keep dims for broadcasting
        mean_X = xp.mean(X_arr, axis=1, keepdims=True)
        # Population variance: divide by N
        n = n_cols
        var_X = xp.sum((X_arr - mean_X) ** 2, axis=1, keepdims=True) / n
        std_X = xp.sqrt(var_X)

        # Avoid division by zero
        std_X = xp.where(std_X < 1e-10, 1.0, std_X)
        result = (X_arr - mean_X) / std_X

        # Set constant rows to zero
        constant_mask = (var_X < 1e-20).ravel()
        if xp.any(constant_mask):
            result[constant_mask, :] = 0.0

    else:  # axis == 0
        # Standardize each column (cells x genes format)
        mean_X = xp.mean(X_arr, axis=0, keepdims=True)
        n = n_rows
        var_X = xp.sum((X_arr - mean_X) ** 2, axis=0, keepdims=True) / n
        std_X = xp.sqrt(var_X)

        std_X = xp.where(std_X < 1e-10, 1.0, std_X)
        result = (X_arr - mean_X) / std_X

        constant_mask = (var_X < 1e-20).ravel()
        if xp.any(constant_mask):
            result[:, constant_mask] = 0.0

    return ensure_numpy(result)


def normalize_lognorm(
    counts,
    scale_factor: float = 10000,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Library size normalization followed by log transformation.

    Computes: log1p(counts / lib_size * scale_factor)

    Parameters
    ----------
    counts : array-like
        Raw count matrix (genes x cells).
    scale_factor : float, default=10000
        Scale factor for normalization.
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    np.ndarray
        Log-normalized expression matrix.

    Examples
    --------
    >>> counts = np.random.poisson(10, size=(100, 1000))
    >>> norm = normalize_lognorm(counts, scale_factor=10000)
    """
    xp = get_array_module(use_gpu)
    counts_arr = xp.asarray(counts, dtype=xp.float64)

    # Library sizes (sum per column/cell)
    lib_sizes = xp.sum(counts_arr, axis=0, keepdims=True)

    # Avoid division by zero
    lib_sizes = xp.where(lib_sizes == 0, 1.0, lib_sizes)

    # Normalize and log transform
    norm_counts = counts_arr / lib_sizes * scale_factor
    result = xp.log1p(norm_counts)

    return ensure_numpy(result)
