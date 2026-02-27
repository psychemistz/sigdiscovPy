"""
ROC/PR evaluation for spatial signal discovery.

Ports the core logic from sim_parse10m/roc_pr_twostage2.py into a
reusable module. Computes gene-level statistics, composite scores,
and evaluates detection performance via AUROC/AUPRC.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score


# ============================================================================
# UTILITIES
# ============================================================================

def smooth_curve(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth a 1D array with a uniform filter."""
    if len(values) < window:
        return values
    return uniform_filter1d(values.astype(float), size=window, mode="nearest")


# ============================================================================
# GENE STATISTICS
# ============================================================================

def compute_gene_statistics(
    ind_df: pd.DataFrame,
    near_radius: float = 1000.0,
    dist_radius: float = 3000.0,
    smooth_window: int = 5,
) -> pd.DataFrame:
    """
    Compute gene-level statistics from I_ND results.

    For each gene, computes:
    - Raw aggregations (mean, std, min, max I_ND)
    - Direction fractions (positive/negative near-field)
    - Smoothed curve metrics (near/far field means, R_Decay, SNR)
    - Composite scores (up_score, down_score, any_score)
    - Cascade scores (cascade_up_wavg, cascade_down_wavg, cascade_any_wavg)

    Parameters
    ----------
    ind_df : pd.DataFrame
        I_ND results with columns: gene, radius, I_ND.
    near_radius : float
        Maximum radius for near-field (default: 1000).
    dist_radius : float
        Minimum radius for far-field (default: 3000).
    smooth_window : int
        Smoothing window size (default: 5).

    Returns
    -------
    pd.DataFrame
        Gene-level statistics with composite scores.
    """
    df = ind_df.copy()
    df["abs_I_ND"] = df["I_ND"].abs()

    near_mask = df["radius"] <= near_radius
    dist_mask = df["radius"] >= dist_radius

    # Raw aggregations
    overall = df.groupby("gene").agg(
        {"I_ND": ["mean", "std", "min", "max"], "abs_I_ND": ["mean", "max"]}
    )
    overall.columns = ["_".join(c) for c in overall.columns]
    overall = overall.rename(
        columns={
            "I_ND_mean": "mean_ind",
            "I_ND_std": "std_ind",
            "I_ND_min": "min_ind",
            "I_ND_max": "max_ind",
            "abs_I_ND_mean": "mean_abs_ind",
            "abs_I_ND_max": "max_abs_ind",
        }
    )

    # Direction fractions
    pos_near = df[near_mask & (df["I_ND"] > 0)].groupby("gene").size()
    neg_near = df[near_mask & (df["I_ND"] < 0)].groupby("gene").size()
    total_near = df[near_mask].groupby("gene").size()

    pos_frac = (pos_near / total_near).rename("pos_frac_near")
    neg_frac = (neg_near / total_near).rename("neg_frac_near")

    stats_df = overall.join(pos_frac, how="left").join(neg_frac, how="left").reset_index()
    stats_df["pos_frac_near"] = stats_df["pos_frac_near"].fillna(0)
    stats_df["neg_frac_near"] = stats_df["neg_frac_near"].fillna(0)

    # Smoothed metrics
    def _compute_smoothed(group):
        group = group.sort_values("radius")
        r = group["radius"].values
        v = group["I_ND"].values
        valid = ~np.isnan(v)

        result = {
            "near_mean_smooth": np.nan,
            "near_abs_mean_smooth": np.nan,
            "dist_abs_mean_smooth": np.nan,
            "R_Decay": np.nan,
            "ind_range_smooth": np.nan,
            "abs_max_smooth": np.nan,
            "SNR": np.nan,
        }

        if valid.sum() < 5:
            return pd.Series(result)

        r_v, v_v = r[valid], v[valid]
        v_s = smooth_curve(v_v, smooth_window)
        abs_s = np.abs(v_s)

        near_m = r_v <= near_radius
        dist_m = r_v >= dist_radius

        near_mean = np.mean(v_s[near_m]) if near_m.sum() > 0 else np.nan
        near_abs = np.mean(abs_s[near_m]) if near_m.sum() > 0 else np.nan
        dist_abs = np.mean(abs_s[dist_m]) if dist_m.sum() > 0 else np.nan
        dist_std = np.std(abs_s[dist_m]) if dist_m.sum() > 2 else np.nan

        R_Decay = (near_abs - dist_abs) / near_abs if near_abs > 1e-10 else np.nan
        ind_range = np.max(v_s) - np.min(v_s)
        abs_max = np.max(abs_s)
        SNR = near_abs / dist_std if dist_std > 1e-10 else np.nan

        result.update(
            {
                "near_mean_smooth": near_mean,
                "near_abs_mean_smooth": near_abs,
                "dist_abs_mean_smooth": dist_abs,
                "R_Decay": R_Decay,
                "ind_range_smooth": ind_range,
                "abs_max_smooth": abs_max,
                "SNR": SNR,
            }
        )

        return pd.Series(result)

    smoothed = df.groupby("gene").apply(_compute_smoothed)
    stats_df = stats_df.merge(smoothed.reset_index(), on="gene", how="left")

    # Direction scores
    dec_score = stats_df["R_Decay"].clip(lower=0, upper=1).fillna(0)

    sig_up = (stats_df["near_mean_smooth"].clip(lower=0) / 0.3).clip(upper=1).fillna(0)
    stats_df["up_score"] = (sig_up + dec_score + stats_df["pos_frac_near"]) / 3

    sig_down = (-stats_df["near_mean_smooth"].clip(upper=0) / 0.3).clip(upper=1).fillna(0)
    stats_df["down_score"] = (sig_down + dec_score + stats_df["neg_frac_near"]) / 3

    sig_any = (stats_df["near_abs_mean_smooth"] / 0.3).clip(upper=1).fillna(0)
    stats_df["any_score"] = (sig_any + dec_score) / 2

    # Cascade scores
    range_99 = stats_df["ind_range_smooth"].quantile(0.99)
    stats_df["ind_range_norm"] = (
        (stats_df["ind_range_smooth"] / range_99).clip(upper=1).fillna(0)
        if range_99 > 0
        else 0
    )

    near_abs_99 = stats_df["near_abs_mean_smooth"].quantile(0.99)
    stats_df["near_abs_norm"] = (
        (stats_df["near_abs_mean_smooth"] / near_abs_99).clip(upper=1).fillna(0)
        if near_abs_99 > 0
        else 0
    )

    stats_df["up_score_norm"] = stats_df["up_score"].clip(lower=0, upper=1).fillna(0)
    stats_df["neg_frac_norm"] = stats_df["neg_frac_near"].clip(lower=0, upper=1).fillna(0)

    # Cascade combinations
    stats_df["cascade_up_mult"] = stats_df["ind_range_norm"] * stats_df["up_score_norm"]
    stats_df["cascade_up_wavg"] = (
        0.6 * stats_df["ind_range_norm"] + 0.4 * stats_df["up_score_norm"]
    )

    stats_df["cascade_down_mult"] = stats_df["ind_range_norm"] * stats_df["neg_frac_norm"]
    stats_df["cascade_down_wavg"] = (
        0.6 * stats_df["ind_range_norm"] + 0.4 * stats_df["neg_frac_norm"]
    )

    stats_df["cascade_any_mult"] = stats_df["ind_range_norm"] * stats_df["near_abs_norm"]
    stats_df["cascade_any_wavg"] = (
        0.6 * stats_df["ind_range_norm"] + 0.4 * stats_df["near_abs_norm"]
    )

    return stats_df


# ============================================================================
# GROUND TRUTH
# ============================================================================

def prepare_ground_truth(
    de_df: pd.DataFrame,
    log2fc_threshold: float = 0.5,
    pval_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Prepare ground truth labels from DE results.

    Parameters
    ----------
    de_df : pd.DataFrame
        DE results with gene, log2fc, and adjusted p-value columns.
    log2fc_threshold : float
        Log2 fold change threshold.
    pval_threshold : float
        Adjusted p-value threshold.

    Returns
    -------
    pd.DataFrame
        Ground truth with columns: gene, log2fc, is_de_up, is_de_down, is_de_any.
    """
    gene_col = next(
        (c for c in ["names", "gene", "Gene", "symbol"] if c in de_df.columns), None
    )
    fc_col = next(
        (c for c in ["log2fc", "log2FoldChange", "logFC"] if c in de_df.columns), None
    )
    pval_col = next(
        (
            c
            for c in ["pval_adj", "padj", "FDR", "p_val_adj"]
            if c in de_df.columns
        ),
        None,
    )

    if gene_col is None or fc_col is None or pval_col is None:
        raise ValueError(
            f"Could not find required columns. Found: {list(de_df.columns)}"
        )

    sig = de_df[pval_col] <= pval_threshold
    up = (de_df[fc_col] >= log2fc_threshold) & sig
    down = (de_df[fc_col] <= -log2fc_threshold) & sig

    return pd.DataFrame(
        {
            "gene": de_df[gene_col],
            "log2fc": de_df[fc_col],
            "is_de_up": up.astype(int),
            "is_de_down": down.astype(int),
            "is_de_any": (up | down).astype(int),
        }
    )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_condition(
    stats_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    """
    Evaluate detection metrics for a condition.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Gene statistics from compute_gene_statistics().
    truth_df : pd.DataFrame
        Ground truth from prepare_ground_truth().
    direction : str
        One of 'up', 'down', 'any'.

    Returns
    -------
    pd.DataFrame
        Evaluation results with AUROC, AUPRC, and lift metrics.
    """
    label = f"is_de_{direction}"
    merged = stats_df.merge(truth_df[["gene", label]], on="gene", how="inner")
    y_true = merged[label].values

    n_pos = y_true.sum()
    n_total = len(y_true)
    if n_pos == 0 or n_pos == n_total:
        return pd.DataFrame()

    roc_baseline = 0.5
    pr_baseline = n_pos / n_total

    if direction == "up":
        metrics = [
            "ind_range_smooth",
            "up_score",
            "near_mean_smooth",
            "pos_frac_near",
            "cascade_up_mult",
            "cascade_up_wavg",
        ]
    elif direction == "down":
        metrics = [
            "ind_range_smooth",
            "neg_frac_near",
            "down_score",
            "cascade_down_mult",
            "cascade_down_wavg",
        ]
    else:
        metrics = [
            "ind_range_smooth",
            "near_abs_mean_smooth",
            "max_abs_ind",
            "any_score",
            "cascade_any_mult",
            "cascade_any_wavg",
        ]

    results = []
    for m in metrics:
        if m not in merged.columns:
            continue

        y_score = merged[m].values
        valid = ~np.isnan(y_score)
        if valid.sum() < 10:
            continue

        try:
            auroc = roc_auc_score(y_true[valid], y_score[valid])
            auprc = average_precision_score(y_true[valid], y_score[valid])

            auc_lift = auroc / roc_baseline
            pr_lift = auprc / pr_baseline if pr_baseline > 0 else np.nan

            results.append(
                {
                    "metric": m,
                    "roc_auc": auroc,
                    "auc_baseline": roc_baseline,
                    "auc_lift": auc_lift,
                    "pr_auc": auprc,
                    "pr_baseline": pr_baseline,
                    "pr_lift": pr_lift,
                    "n_samples": int(valid.sum()),
                    "n_positive": int(y_true[valid].sum()),
                }
            )
        except Exception:
            pass

    return pd.DataFrame(results)


def evaluate_all_directions(
    stats_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate detection metrics for all directions (up, down, any).

    Parameters
    ----------
    stats_df : pd.DataFrame
        Gene statistics from compute_gene_statistics().
    truth_df : pd.DataFrame
        Ground truth from prepare_ground_truth().

    Returns
    -------
    pd.DataFrame
        Combined evaluation results across all directions.
    """
    all_results = []
    for direction in ["up", "down", "any"]:
        result = evaluate_condition(stats_df, truth_df, direction)
        if len(result) > 0:
            result["direction"] = direction
            all_results.append(result)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def compute_roc_pr_curves(
    stats_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    metric: str,
    direction: str,
) -> Optional[dict]:
    """
    Compute ROC and PR curves for a specific metric and direction.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Gene statistics.
    truth_df : pd.DataFrame
        Ground truth.
    metric : str
        Score column name in stats_df.
    direction : str
        One of 'up', 'down', 'any'.

    Returns
    -------
    dict or None
        Dictionary with fpr, tpr, precision, recall, roc_auc, pr_auc.
        None if insufficient data.
    """
    from sklearn.metrics import precision_recall_curve, roc_curve

    label = f"is_de_{direction}"
    merged = stats_df.merge(truth_df[["gene", label]], on="gene", how="inner")

    if metric not in merged.columns:
        return None

    y_true = merged[label].values
    y_score = merged[metric].values
    valid = ~np.isnan(y_score)

    if valid.sum() < 10 or y_true[valid].sum() == 0:
        return None

    try:
        fpr, tpr, roc_thresholds = roc_curve(y_true[valid], y_score[valid])
        prec, rec, pr_thresholds = precision_recall_curve(y_true[valid], y_score[valid])
        auroc = roc_auc_score(y_true[valid], y_score[valid])
        auprc = average_precision_score(y_true[valid], y_score[valid])

        return {
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
            "precision": prec,
            "recall": rec,
            "pr_thresholds": pr_thresholds,
            "roc_auc": auroc,
            "pr_auc": auprc,
            "n_samples": int(valid.sum()),
            "n_positive": int(y_true[valid].sum()),
        }
    except Exception:
        return None
