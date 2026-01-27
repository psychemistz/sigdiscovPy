"""
Permutation testing for spatial statistics.

GPU-accelerated permutation tests for Moran's I and I_ND.
"""

from typing import Literal, Optional

import numpy as np

from sigdiscovpy.core.metrics import compute_ind_from_lag, compute_moran_from_lag
from sigdiscovpy.gpu.backend import GPU_AVAILABLE, ensure_numpy


def permutation_test(
    z_f,
    lag_g,
    n_permutations: int = 999,
    metric: Literal["moran", "ind"] = "ind",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    random_seed: Optional[int] = None,
    use_gpu: bool = True,
) -> dict:
    """
    Permutation test for spatial correlation significance.

    Parameters
    ----------
    z_f : array-like
        Standardized factor expression (length n).
    lag_g : array-like
        Spatial lag of gene expression (length n).
    n_permutations : int, default=999
        Number of permutations.
    metric : {"moran", "ind"}, default="ind"
        Metric to test.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        Alternative hypothesis.
    random_seed : int, optional
        Random seed for reproducibility.
    use_gpu : bool, default=True
        Use GPU acceleration if available.

    Returns
    -------
    dict
        Dictionary with:
        - 'observed': Observed statistic
        - 'pvalue': Permutation p-value
        - 'null_mean': Mean of null distribution
        - 'null_std': Std of null distribution
        - 'z_score': Z-score of observed value

    Examples
    --------
    >>> z_f = np.random.randn(100)
    >>> lag_g = np.random.randn(100)
    >>> result = permutation_test(z_f, lag_g, n_permutations=999)
    >>> result['pvalue']  # Should be around 0.5 for random data
    """
    z_f = np.asarray(z_f, dtype=np.float64)
    lag_g = np.asarray(lag_g, dtype=np.float64)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Compute observed statistic
    if metric == "moran":
        observed = compute_moran_from_lag(z_f, lag_g, use_gpu=use_gpu)
    else:
        observed = compute_ind_from_lag(z_f, lag_g, use_gpu=use_gpu)

    if np.isnan(observed):
        return {
            "observed": np.nan,
            "pvalue": np.nan,
            "null_mean": np.nan,
            "null_std": np.nan,
            "z_score": np.nan,
        }

    # GPU-accelerated permutation test
    if use_gpu and GPU_AVAILABLE:
        null_distribution = _permutation_test_gpu(z_f, lag_g, n_permutations, metric)
    else:
        null_distribution = _permutation_test_cpu(z_f, lag_g, n_permutations, metric)

    # Compute p-value
    if alternative == "two-sided":
        pvalue = (np.sum(np.abs(null_distribution) >= np.abs(observed)) + 1) / (n_permutations + 1)
    elif alternative == "greater":
        pvalue = (np.sum(null_distribution >= observed) + 1) / (n_permutations + 1)
    else:  # less
        pvalue = (np.sum(null_distribution <= observed) + 1) / (n_permutations + 1)

    null_mean = np.mean(null_distribution)
    null_std = np.std(null_distribution)
    z_score = (observed - null_mean) / null_std if null_std > 1e-10 else 0.0

    return {
        "observed": observed,
        "pvalue": pvalue,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
    }


def _permutation_test_cpu(
    z_f: np.ndarray,
    lag_g: np.ndarray,
    n_permutations: int,
    metric: str,
) -> np.ndarray:
    """CPU implementation of permutation test."""
    n = len(z_f)
    null_distribution = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Permute factor expression
        perm_idx = np.random.permutation(n)
        z_f_perm = z_f[perm_idx]

        if metric == "moran":
            null_distribution[i] = np.dot(z_f_perm, lag_g) / n
        else:
            norm_f = np.linalg.norm(z_f_perm)
            norm_lag = np.linalg.norm(lag_g)
            if norm_f > 1e-10 and norm_lag > 1e-10:
                null_distribution[i] = np.dot(z_f_perm, lag_g) / (norm_f * norm_lag)
            else:
                null_distribution[i] = np.nan

    return null_distribution


def _permutation_test_gpu(
    z_f: np.ndarray,
    lag_g: np.ndarray,
    n_permutations: int,
    metric: str,
) -> np.ndarray:
    """GPU-accelerated permutation test."""
    import cupy as cp

    n = len(z_f)
    z_f_gpu = cp.asarray(z_f, dtype=cp.float64)
    lag_g_gpu = cp.asarray(lag_g, dtype=cp.float64)

    # Generate all permutations at once
    perm_indices = cp.array([cp.random.permutation(n) for _ in range(n_permutations)])

    # Batch permute: (n_permutations, n)
    z_f_perms = z_f_gpu[perm_indices]

    # Batch dot products
    dots = cp.sum(z_f_perms * lag_g_gpu, axis=1)

    if metric == "moran":
        null_distribution = dots / n
    else:
        # I_ND: need norms
        perm_norms = cp.linalg.norm(z_f_perms, axis=1)
        lag_norm = cp.linalg.norm(lag_g_gpu)

        denominator = perm_norms * lag_norm
        null_distribution = cp.where(
            denominator > 1e-10,
            dots / denominator,
            cp.nan,
        )

    return ensure_numpy(null_distribution)


def batch_permutation_test(
    z_f,
    lag_G,
    n_permutations: int = 999,
    metric: Literal["moran", "ind"] = "ind",
    random_seed: Optional[int] = None,
    use_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch permutation test for one factor against all genes.

    Parameters
    ----------
    z_f : array-like
        Standardized factor expression (length n).
    lag_G : array-like
        Spatial lag matrix (n x n_genes).
    n_permutations : int, default=999
        Number of permutations.
    metric : {"moran", "ind"}, default="ind"
        Metric to test.
    random_seed : int, optional
        Random seed.
    use_gpu : bool, default=True
        Use GPU acceleration.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - observed: Observed statistics (n_genes,)
        - pvalues: P-values (n_genes,)

    Examples
    --------
    >>> n, n_genes = 100, 50
    >>> z_f = np.random.randn(n)
    >>> lag_G = np.random.randn(n, n_genes)
    >>> observed, pvalues = batch_permutation_test(z_f, lag_G, n_permutations=999)
    """
    z_f = np.asarray(z_f, dtype=np.float64)
    lag_G = np.asarray(lag_G, dtype=np.float64)

    if random_seed is not None:
        np.random.seed(random_seed)

    if use_gpu and GPU_AVAILABLE:
        return _batch_permutation_test_gpu(z_f, lag_G, n_permutations, metric)
    else:
        return _batch_permutation_test_cpu(z_f, lag_G, n_permutations, metric)


def _batch_permutation_test_cpu(
    z_f: np.ndarray,
    lag_G: np.ndarray,
    n_permutations: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU batch permutation test."""
    n = len(z_f)
    n_genes = lag_G.shape[1]

    # Compute observed values
    dots = z_f @ lag_G  # (n_genes,)

    if metric == "moran":
        observed = dots / n
    else:
        norm_f = np.linalg.norm(z_f)
        lag_norms = np.linalg.norm(lag_G, axis=0)
        denominator = norm_f * lag_norms
        observed = np.where(denominator > 1e-10, dots / denominator, np.nan)

    # Permutation null distribution
    null_counts = np.zeros(n_genes)

    for _ in range(n_permutations):
        perm_idx = np.random.permutation(n)
        z_f_perm = z_f[perm_idx]

        perm_dots = z_f_perm @ lag_G

        if metric == "moran":
            perm_stats = perm_dots / n
        else:
            perm_norm = np.linalg.norm(z_f_perm)
            denominator = perm_norm * lag_norms
            perm_stats = np.where(denominator > 1e-10, perm_dots / denominator, np.nan)

        null_counts += np.abs(perm_stats) >= np.abs(observed)

    pvalues = (null_counts + 1) / (n_permutations + 1)

    return observed, pvalues


def _batch_permutation_test_gpu(
    z_f: np.ndarray,
    lag_G: np.ndarray,
    n_permutations: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU batch permutation test."""
    import cupy as cp

    n = len(z_f)
    n_genes = lag_G.shape[1]

    z_f_gpu = cp.asarray(z_f, dtype=cp.float64)
    lag_G_gpu = cp.asarray(lag_G, dtype=cp.float64)

    # Compute observed values
    dots = z_f_gpu @ lag_G_gpu

    if metric == "moran":
        observed = dots / n
    else:
        norm_f = cp.linalg.norm(z_f_gpu)
        lag_norms = cp.linalg.norm(lag_G_gpu, axis=0)
        denominator = norm_f * lag_norms
        observed = cp.where(denominator > 1e-10, dots / denominator, cp.nan)

    # Batch permutation test
    null_counts = cp.zeros(n_genes)

    # Process in batches to manage memory
    batch_size = min(100, n_permutations)

    for batch_start in range(0, n_permutations, batch_size):
        batch_end = min(batch_start + batch_size, n_permutations)
        n_batch = batch_end - batch_start

        # Generate permutation indices
        perm_indices = cp.array([cp.random.permutation(n) for _ in range(n_batch)])
        z_f_perms = z_f_gpu[perm_indices]  # (n_batch, n)

        # Batch dot products: (n_batch, n) @ (n, n_genes) = (n_batch, n_genes)
        perm_dots = z_f_perms @ lag_G_gpu

        if metric == "moran":
            perm_stats = perm_dots / n
        else:
            perm_norms = cp.linalg.norm(z_f_perms, axis=1, keepdims=True)
            denominator = perm_norms * lag_norms.reshape(1, -1)
            perm_stats = cp.where(denominator > 1e-10, perm_dots / denominator, cp.nan)

        # Count exceedances
        null_counts += cp.sum(cp.abs(perm_stats) >= cp.abs(observed), axis=0)

    pvalues = (null_counts + 1) / (n_permutations + 1)

    return ensure_numpy(observed), ensure_numpy(pvalues)
