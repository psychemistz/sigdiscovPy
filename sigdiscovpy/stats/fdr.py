"""
FDR correction methods.

Implements:
- Benjamini-Hochberg (BH)
- Benjamini-Yekutieli (BY)
- Bonferroni
"""

from typing import Literal
import numpy as np


def apply_fdr_correction(
    pvalues,
    method: Literal["bh", "by", "bonferroni"] = "bh",
    alpha: float = 0.05,
) -> dict:
    """
    Apply FDR correction to p-values.

    Parameters
    ----------
    pvalues : array-like
        Array of p-values.
    method : {"bh", "by", "bonferroni"}, default="bh"
        Correction method:
        - "bh": Benjamini-Hochberg (controls FDR)
        - "by": Benjamini-Yekutieli (controls FDR under dependency)
        - "bonferroni": Bonferroni (controls FWER)
    alpha : float, default=0.05
        Significance threshold after correction.

    Returns
    -------
    dict
        Dictionary with:
        - 'adjusted_pvalues': Corrected p-values
        - 'significant': Boolean mask of significant tests
        - 'n_significant': Number of significant tests
        - 'method': Method used
        - 'alpha': Alpha threshold used

    Examples
    --------
    >>> pvalues = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
    >>> result = apply_fdr_correction(pvalues, method="bh")
    >>> result['n_significant']
    2
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = len(pvalues)

    if n == 0:
        return {
            "adjusted_pvalues": np.array([]),
            "significant": np.array([], dtype=bool),
            "n_significant": 0,
            "method": method,
            "alpha": alpha,
        }

    # Handle NaN values
    nan_mask = np.isnan(pvalues)
    valid_pvalues = pvalues[~nan_mask]
    n_valid = len(valid_pvalues)

    if n_valid == 0:
        adjusted = np.full(n, np.nan)
        return {
            "adjusted_pvalues": adjusted,
            "significant": np.zeros(n, dtype=bool),
            "n_significant": 0,
            "method": method,
            "alpha": alpha,
        }

    if method == "bonferroni":
        adjusted_valid = np.minimum(valid_pvalues * n_valid, 1.0)

    elif method == "bh":
        # Benjamini-Hochberg
        sorted_idx = np.argsort(valid_pvalues)
        sorted_pvalues = valid_pvalues[sorted_idx]

        # Compute adjusted p-values
        ranks = np.arange(1, n_valid + 1)
        adjusted_sorted = sorted_pvalues * n_valid / ranks

        # Enforce monotonicity (cumulative minimum from the end)
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

        # Unsort
        adjusted_valid = np.empty(n_valid)
        adjusted_valid[sorted_idx] = adjusted_sorted

    elif method == "by":
        # Benjamini-Yekutieli
        sorted_idx = np.argsort(valid_pvalues)
        sorted_pvalues = valid_pvalues[sorted_idx]

        # BY correction factor: sum(1/i) for i in 1..n
        c_n = np.sum(1.0 / np.arange(1, n_valid + 1))

        ranks = np.arange(1, n_valid + 1)
        adjusted_sorted = sorted_pvalues * n_valid * c_n / ranks

        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

        adjusted_valid = np.empty(n_valid)
        adjusted_valid[sorted_idx] = adjusted_sorted

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'bh', 'by', or 'bonferroni'.")

    # Reconstruct full array with NaN values
    adjusted = np.full(n, np.nan)
    adjusted[~nan_mask] = adjusted_valid

    significant = adjusted < alpha

    return {
        "adjusted_pvalues": adjusted,
        "significant": significant,
        "n_significant": int(np.sum(significant)),
        "method": method,
        "alpha": alpha,
    }


def compute_qvalues(pvalues) -> np.ndarray:
    """
    Compute q-values (Storey's method).

    Parameters
    ----------
    pvalues : array-like
        Array of p-values.

    Returns
    -------
    np.ndarray
        Q-values.
    """
    # Simple BH-based q-value estimation
    result = apply_fdr_correction(pvalues, method="bh")
    return result["adjusted_pvalues"]
