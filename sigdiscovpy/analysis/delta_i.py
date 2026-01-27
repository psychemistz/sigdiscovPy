"""
Delta I computation for identifying distance-dependent spatial interactions.
"""

from typing import Optional

import numpy as np


def compute_delta_i(
    metrics_by_radius: dict[float, np.ndarray],
    radii: list[float],
    baseline_radius: Optional[float] = None,
) -> np.ndarray:
    """
    Compute delta I (change in metric from baseline to peak).

    Parameters
    ----------
    metrics_by_radius : dict
        Dictionary mapping radius to metrics array.
    radii : list
        List of radii (sorted).
    baseline_radius : float, optional
        Radius to use as baseline. Default: smallest radius.

    Returns
    -------
    np.ndarray
        Delta I values (same shape as individual metric arrays).

    Examples
    --------
    >>> metrics = {50: np.random.randn(10, 100), 100: np.random.randn(10, 100)}
    >>> delta = compute_delta_i(metrics, [50, 100])
    >>> delta.shape
    (10, 100)
    """
    radii = sorted(radii)

    if baseline_radius is None:
        baseline_radius = radii[0]

    baseline = metrics_by_radius[baseline_radius]

    # Stack all metrics and find peak
    all_metrics = np.stack([metrics_by_radius[r] for r in radii], axis=0)

    # Find peak across radii (handle NaN)
    peak = np.nanmax(all_metrics, axis=0)

    return peak - baseline


def compute_delta_i_profile(
    metrics_by_radius: dict[float, np.ndarray],
    radii: list[float],
    gene_idx: int,
    factor_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get metric profile across radii for a specific gene-factor pair.

    Parameters
    ----------
    metrics_by_radius : dict
        Dictionary mapping radius to metrics matrix.
    radii : list
        List of radii.
    gene_idx : int
        Target gene index.
    factor_idx : int, default=0
        Factor gene index.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (radii_array, metrics_array) for plotting.
    """
    radii_sorted = np.array(sorted(radii))
    metrics = np.array([metrics_by_radius[r][factor_idx, gene_idx] for r in radii_sorted])

    return radii_sorted, metrics


def find_peak_radius(
    metrics_by_radius: dict[float, np.ndarray],
    radii: list[float],
    factor_idx: int = 0,
) -> np.ndarray:
    """
    Find the radius at which each gene reaches peak metric.

    Parameters
    ----------
    metrics_by_radius : dict
        Dictionary mapping radius to metrics matrix.
    radii : list
        List of radii.
    factor_idx : int, default=0
        Factor gene index.

    Returns
    -------
    np.ndarray
        Peak radius for each gene.
    """
    radii_sorted = sorted(radii)

    # Stack metrics: (n_radii, n_genes)
    all_metrics = np.stack([metrics_by_radius[r][factor_idx, :] for r in radii_sorted], axis=0)

    # Find argmax for each gene
    peak_idx = np.nanargmax(all_metrics, axis=0)
    peak_radii = np.array([radii_sorted[i] for i in peak_idx])

    return peak_radii


def classify_interaction_type(
    metrics_by_radius: dict[float, np.ndarray],
    radii: list[float],
    gene_idx: int,
    factor_idx: int = 0,
    threshold: float = 0.1,
) -> str:
    """
    Classify spatial interaction type based on distance profile.

    Parameters
    ----------
    metrics_by_radius : dict
        Dictionary mapping radius to metrics matrix.
    radii : list
        List of radii.
    gene_idx : int
        Target gene index.
    factor_idx : int, default=0
        Factor gene index.
    threshold : float, default=0.1
        Minimum metric value to consider significant.

    Returns
    -------
    str
        Interaction type: "short-range", "long-range", "bidirectional", or "none"
    """
    radii_sorted, metrics = compute_delta_i_profile(metrics_by_radius, radii, gene_idx, factor_idx)

    # Mask NaN values
    valid = ~np.isnan(metrics)
    if not np.any(valid):
        return "none"

    metrics_valid = metrics[valid]
    radii_valid = radii_sorted[valid]

    max_metric = np.max(np.abs(metrics_valid))
    if max_metric < threshold:
        return "none"

    # Find peak location
    peak_idx = np.argmax(np.abs(metrics_valid))
    peak_radius = radii_valid[peak_idx]

    # Classify based on peak location relative to radius range
    radius_range = radii_valid[-1] - radii_valid[0]
    relative_peak = (peak_radius - radii_valid[0]) / radius_range

    if relative_peak < 0.33:
        return "short-range"
    elif relative_peak > 0.67:
        return "long-range"
    else:
        return "medium-range"
