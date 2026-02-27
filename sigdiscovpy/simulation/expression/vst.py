"""
Variance Stabilizing Transform (VST) functions for simulation.

Ports 3 VST transforms from reference core.py:
- log1p: log1p with centering
- pearson: Pearson residuals (sctransform-style)
- shifted_log: shifted log ensuring non-expressors get negative values
"""

import numpy as np


def apply_vst_log1p(raw_expr: np.ndarray) -> np.ndarray:
    """Log1p VST normalization with centering.

    Applies log1p transform then centers by global mean.
    Non-expressing cells get small negative values.

    Parameters
    ----------
    raw_expr : np.ndarray
        Raw expression values.

    Returns
    -------
    np.ndarray
        VST-normalized expression.
    """
    log_expr = np.log1p(raw_expr)
    return log_expr - np.mean(log_expr)


def apply_vst_pearson(raw_expr: np.ndarray, theta: float = 100, clip_value: float = 10) -> np.ndarray:
    """Pearson residual VST normalization (sctransform-style).

    Parameters
    ----------
    raw_expr : np.ndarray
        Raw expression values.
    theta : float
        Dispersion parameter for negative binomial model.
    clip_value : float
        Clip extreme residuals.

    Returns
    -------
    np.ndarray
        Pearson residuals.
    """
    mu = np.mean(raw_expr) + 1e-10
    variance = mu + (mu ** 2) / theta
    residuals = (raw_expr - mu) / np.sqrt(variance)
    return np.clip(residuals, -clip_value, clip_value)


def apply_vst_shifted_log(raw_expr: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
    """Shifted log normalization.

    Ensures non-expressors get negative values.
    raw=0 maps to -0.5.

    Parameters
    ----------
    raw_expr : np.ndarray
        Raw expression values.
    pseudocount : float
        Pseudocount added before log.

    Returns
    -------
    np.ndarray
        Shifted log-normalized expression.
    """
    log_expr = np.log(raw_expr + pseudocount)
    baseline = np.log(pseudocount)
    shift = baseline + 0.5
    return log_expr - shift


def apply_vst(raw_expr: np.ndarray, method: str) -> np.ndarray:
    """Dispatch to appropriate VST method.

    Parameters
    ----------
    raw_expr : np.ndarray
        Raw expression values.
    method : str
        VST method: 'log1p', 'pearson', or 'shifted_log'.

    Returns
    -------
    np.ndarray
        VST-normalized expression.
    """
    if method == "log1p":
        return apply_vst_log1p(raw_expr)
    elif method == "pearson":
        return apply_vst_pearson(raw_expr)
    elif method == "shifted_log":
        return apply_vst_shifted_log(raw_expr)
    else:
        raise ValueError(f"Unknown VST method: {method}. Available: log1p, pearson, shifted_log")
