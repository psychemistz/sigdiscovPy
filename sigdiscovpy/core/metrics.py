"""
Core spatial correlation metrics.

Implements:
- Bivariate Moran's I
- I_ND (normalized directional Moran's I / cosine similarity)

These functions operate on pre-computed spatial lags for efficiency.
"""

from typing import Literal

import numpy as np

from sigdiscovpy.gpu.backend import ensure_numpy, get_array_module


def _validate_vectors(z_f, lag_g, func_name: str):
    """
    Validate input vectors for metric computation.

    Parameters
    ----------
    z_f : array-like
        First vector.
    lag_g : array-like
        Second vector.
    func_name : str
        Name of calling function for error messages.

    Raises
    ------
    ValueError
        If inputs are invalid (dimension mismatch, empty, or contain NaN/Inf).
    """
    z_f = np.asarray(z_f)
    lag_g = np.asarray(lag_g)

    # Check for empty arrays
    if z_f.size == 0 or lag_g.size == 0:
        raise ValueError(f"{func_name}: Input vectors cannot be empty")

    # Check dimension match
    if z_f.shape[0] != lag_g.shape[0]:
        raise ValueError(
            f"{func_name}: Dimension mismatch - z_f has length {z_f.shape[0]}, "
            f"lag_g has length {lag_g.shape[0]}"
        )

    # Check for NaN/Inf
    if np.any(~np.isfinite(z_f)):
        raise ValueError(f"{func_name}: z_f contains NaN or Inf values")
    if np.any(~np.isfinite(lag_g)):
        raise ValueError(f"{func_name}: lag_g contains NaN or Inf values")


def compute_moran_from_lag(
    z_f,
    lag_g,
    use_gpu: bool = True,
) -> float:
    """
    Compute Bivariate Moran's I from pre-computed spatial lag.

    Formula: I = z_f' * lag_g / n

    Parameters
    ----------
    z_f : array-like
        Standardized factor expression vector (length n).
    lag_g : array-like
        Pre-computed spatial lag of gene expression (W * z_g), same length as z_f.
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    float
        Moran's I value (unbounded).

    Examples
    --------
    >>> z_f = np.array([1.0, -1.0, 1.0, -1.0])
    >>> lag_g = np.array([0.5, -0.5, 0.5, -0.5])
    >>> I = compute_moran_from_lag(z_f, lag_g)
    >>> I  # Should be 0.5
    0.5

    Notes
    -----
    Positive I indicates spatial clustering (similar values near each other).
    Negative I indicates spatial dispersion (dissimilar values near each other).

    Raises
    ------
    ValueError
        If inputs have mismatched dimensions, are empty, or contain NaN/Inf.
    """
    # Validate inputs
    _validate_vectors(z_f, lag_g, "compute_moran_from_lag")

    xp = get_array_module(use_gpu)
    z_f_arr = xp.asarray(z_f, dtype=xp.float64)
    lag_g_arr = xp.asarray(lag_g, dtype=xp.float64)

    n = z_f_arr.shape[0]
    moran_i = float(xp.dot(z_f_arr, lag_g_arr)) / n

    return moran_i


def compute_ind_from_lag(
    z_f,
    lag_g,
    use_gpu: bool = True,
) -> float:
    """
    Compute I_ND (cosine similarity) from pre-computed spatial lag.

    Formula: I_ND = z_f' * lag_g / (||z_f|| * ||lag_g||)

    This is the normalized directional Moran's I, bounded between -1 and 1.

    Parameters
    ----------
    z_f : array-like
        Standardized factor expression vector (length n).
    lag_g : array-like
        Pre-computed spatial lag of gene expression (W * z_g).
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    float
        I_ND value bounded between -1 and 1.
        Returns np.nan if either vector has near-zero norm.

    Examples
    --------
    >>> z_f = np.array([1.0, 0.0, 1.0, 0.0])
    >>> lag_g = np.array([1.0, 0.0, 1.0, 0.0])
    >>> I_ND = compute_ind_from_lag(z_f, lag_g)
    >>> I_ND  # Should be 1.0 (perfect correlation)
    1.0

    Notes
    -----
    I_ND is the cosine of the angle between z_f and the spatial lag of z_g.
    - +1: Perfect positive spatial association
    -  0: No spatial association
    - -1: Perfect negative spatial association

    Raises
    ------
    ValueError
        If inputs have mismatched dimensions, are empty, or contain NaN/Inf.
    """
    # Validate inputs
    _validate_vectors(z_f, lag_g, "compute_ind_from_lag")

    xp = get_array_module(use_gpu)
    z_f_arr = xp.asarray(z_f, dtype=xp.float64)
    lag_g_arr = xp.asarray(lag_g, dtype=xp.float64)

    # Compute norms
    norm_f = float(xp.linalg.norm(z_f_arr))
    norm_lag = float(xp.linalg.norm(lag_g_arr))

    # Check for degenerate cases
    if norm_f < 1e-10 or norm_lag < 1e-10:
        return np.nan

    # Cosine similarity
    I_ND = float(xp.dot(z_f_arr, lag_g_arr)) / (norm_f * norm_lag)

    return I_ND


def compute_metric_batch(
    z_f,
    lag_G,
    metric: Literal["moran", "ind"] = "ind",
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Batch compute metrics for one factor against all genes.

    Efficiently computes Moran's I or I_ND for one factor expression vector
    against a matrix of pre-computed spatial lags for all genes.

    Parameters
    ----------
    z_f : array-like
        Standardized factor expression vector (length n).
    lag_G : array-like
        Spatial lag matrix (n x n_genes), where each column is W * z_g for gene g.
    metric : {"moran", "ind"}, default="ind"
        Metric to compute:
        - "moran": Bivariate Moran's I
        - "ind": I_ND (cosine similarity)
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    np.ndarray
        Vector of metric values (length n_genes).

    Examples
    --------
    >>> n = 100
    >>> n_genes = 50
    >>> z_f = np.random.randn(n)
    >>> lag_G = np.random.randn(n, n_genes)
    >>> metrics = compute_metric_batch(z_f, lag_G, metric="ind")
    >>> metrics.shape
    (50,)

    Notes
    -----
    This is the workhorse function for genome-wide analysis:
    1. Compute spatial lag matrix once: lag_G = W @ Z_g
    2. Call this function to get metrics for all genes in one vectorized operation

    Raises
    ------
    ValueError
        If inputs have mismatched dimensions, are empty, or contain NaN/Inf.
    """
    # Convert to arrays for validation
    z_f_np = np.asarray(z_f)
    lag_G_np = np.asarray(lag_G)

    # Validate inputs
    if z_f_np.size == 0 or lag_G_np.size == 0:
        raise ValueError("compute_metric_batch: Input arrays cannot be empty")

    if z_f_np.shape[0] != lag_G_np.shape[0]:
        raise ValueError(
            f"compute_metric_batch: Dimension mismatch - z_f has length {z_f_np.shape[0]}, "
            f"lag_G has {lag_G_np.shape[0]} rows"
        )

    if np.any(~np.isfinite(z_f_np)):
        raise ValueError("compute_metric_batch: z_f contains NaN or Inf values")
    if np.any(~np.isfinite(lag_G_np)):
        raise ValueError("compute_metric_batch: lag_G contains NaN or Inf values")

    xp = get_array_module(use_gpu)
    z_f_arr = xp.asarray(z_f, dtype=xp.float64)
    lag_G_arr = xp.asarray(lag_G, dtype=xp.float64)

    n = z_f_arr.shape[0]
    n_genes = lag_G_arr.shape[1]

    if metric == "moran":
        # Moran's I: I_g = z_f' * lag_G_g / n
        # Vectorized: result = (z_f @ lag_G) / n
        result = (z_f_arr @ lag_G_arr) / n

    elif metric == "ind":
        # I_ND: I_g = z_f' * lag_G_g / (||z_f|| * ||lag_G_g||)
        norm_f = xp.linalg.norm(z_f_arr)

        if float(norm_f) < 1e-10:
            return np.full(n_genes, np.nan)

        # Normalize factor once
        f_normalized = z_f_arr / norm_f

        # Dot products: f_normalized @ lag_G (shape: n_genes,)
        correlations = f_normalized @ lag_G_arr

        # Column norms of lag_G
        lag_norms = xp.linalg.norm(lag_G_arr, axis=0)

        # Compute I_ND with safe division
        result = xp.where(
            lag_norms > 1e-10,
            correlations / lag_norms,
            xp.nan,
        )

    else:
        raise ValueError(f"Unknown metric: '{metric}'. Use 'moran' or 'ind'.")

    return ensure_numpy(result)


def compute_metrics_matrix(
    Z_f,
    lag_G,
    metric: Literal["moran", "ind"] = "ind",
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Compute metrics matrix for multiple factors against all genes.

    Parameters
    ----------
    Z_f : array-like
        Standardized factor expression matrix (n x n_factors).
    lag_G : array-like
        Spatial lag matrix (n x n_genes).
    metric : {"moran", "ind"}, default="ind"
        Metric to compute.
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    np.ndarray
        Metrics matrix (n_factors x n_genes).
    """
    xp = get_array_module(use_gpu)
    Z_f_arr = xp.asarray(Z_f, dtype=xp.float64)
    lag_G_arr = xp.asarray(lag_G, dtype=xp.float64)

    n = Z_f_arr.shape[0]

    if metric == "moran":
        # Moran's I: result = (Z_f.T @ lag_G) / n
        result = (Z_f_arr.T @ lag_G_arr) / n

    elif metric == "ind":
        # I_ND with column-wise normalization
        # Normalize factors
        f_norms = xp.linalg.norm(Z_f_arr, axis=0, keepdims=True)
        f_norms = xp.where(f_norms < 1e-10, 1.0, f_norms)
        Z_f_normalized = Z_f_arr / f_norms

        # Dot products: (n_factors x n) @ (n x n_genes) = (n_factors x n_genes)
        correlations = Z_f_normalized.T @ lag_G_arr

        # Column norms of lag_G
        lag_norms = xp.linalg.norm(lag_G_arr, axis=0, keepdims=True)

        # Safe division
        result = xp.where(
            lag_norms > 1e-10,
            correlations / lag_norms,
            xp.nan,
        )

        # Set NaN for factors with zero norm
        zero_factor_mask = (xp.linalg.norm(Z_f_arr, axis=0) < 1e-10).reshape(-1, 1)
        result = xp.where(zero_factor_mask, xp.nan, result)

    else:
        raise ValueError(f"Unknown metric: '{metric}'. Use 'moran' or 'ind'.")

    return ensure_numpy(result)
