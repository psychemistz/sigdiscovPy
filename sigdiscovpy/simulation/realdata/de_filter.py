"""
Differential expression filtering for spatial data.

Implements Wilcoxon rank-sum test with FDR correction.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from sigdiscovpy.stats.fdr import apply_fdr_correction


@dataclass
class DEResult:
    """Result of differential expression analysis."""

    gene_names: list[str]
    pvalues: np.ndarray
    adjusted_pvalues: np.ndarray
    log2_fold_change: np.ndarray
    mean_group1: np.ndarray
    mean_group2: np.ndarray
    significant_mask: np.ndarray

    @property
    def n_significant(self) -> int:
        return int(np.sum(self.significant_mask))

    def get_significant_genes(self) -> list[str]:
        """Return list of significant gene names."""
        return [self.gene_names[i] for i in np.where(self.significant_mask)[0]]

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "gene": self.gene_names,
                "pvalue": self.pvalues,
                "adjusted_pvalue": self.adjusted_pvalues,
                "log2_fold_change": self.log2_fold_change,
                "mean_group1": self.mean_group1,
                "mean_group2": self.mean_group2,
                "significant": self.significant_mask,
            }
        )


def wilcoxon_test(
    expr: np.ndarray,
    group1_mask: np.ndarray,
    group2_mask: np.ndarray,
    gene_names: list[str],
    fdr_method: str = "bh",
    fdr_threshold: float = 0.05,
    min_log2fc: float = 0.0,
    pseudocount: float = 1e-9,
) -> DEResult:
    """
    Perform Wilcoxon rank-sum test for differential expression.

    Parameters
    ----------
    expr : np.ndarray
        Expression matrix (n_genes, n_cells).
    group1_mask : np.ndarray
        Boolean mask for group 1 cells.
    group2_mask : np.ndarray
        Boolean mask for group 2 cells.
    gene_names : list of str
        Gene names.
    fdr_method : str
        FDR correction method ('bh', 'by', 'bonferroni').
    fdr_threshold : float
        FDR threshold for significance.
    min_log2fc : float
        Minimum absolute log2 fold change for significance.
    pseudocount : float
        Pseudocount for log2 fold change calculation.

    Returns
    -------
    DEResult
        Differential expression results.
    """
    n_genes = expr.shape[0]

    pvalues = np.zeros(n_genes)
    mean_g1 = np.zeros(n_genes)
    mean_g2 = np.zeros(n_genes)

    expr_g1 = expr[:, group1_mask]
    expr_g2 = expr[:, group2_mask]

    for i in range(n_genes):
        g1_vals = expr_g1[i, :]
        g2_vals = expr_g2[i, :]

        mean_g1[i] = np.mean(g1_vals)
        mean_g2[i] = np.mean(g2_vals)

        # Skip if no variance
        if np.std(g1_vals) == 0 and np.std(g2_vals) == 0:
            pvalues[i] = 1.0
        else:
            try:
                _, pvalues[i] = stats.mannwhitneyu(g1_vals, g2_vals, alternative="two-sided")
            except ValueError:
                pvalues[i] = 1.0

    # Calculate log2 fold change
    log2fc = np.log2((mean_g1 + pseudocount) / (mean_g2 + pseudocount))

    # FDR correction
    fdr_result = apply_fdr_correction(pvalues, method=fdr_method, alpha=fdr_threshold)
    adjusted_pvalues = fdr_result["adjusted_pvalues"]

    # Determine significance
    significant_mask = (adjusted_pvalues < fdr_threshold) & (np.abs(log2fc) >= min_log2fc)

    return DEResult(
        gene_names=gene_names,
        pvalues=pvalues,
        adjusted_pvalues=adjusted_pvalues,
        log2_fold_change=log2fc,
        mean_group1=mean_g1,
        mean_group2=mean_g2,
        significant_mask=significant_mask,
    )


def filter_by_celltype_de(
    expr: np.ndarray,
    cell_types: np.ndarray,
    gene_names: list[str],
    sender_type: str,
    receiver_type: str,
    fdr_threshold: float = 0.05,
    min_log2fc: float = 1.0,
) -> tuple[list[str], list[str]]:
    """
    Filter genes by differential expression between cell types.

    Returns sender-specific and receiver-specific genes.

    Parameters
    ----------
    expr : np.ndarray
        Expression matrix (n_genes, n_cells).
    cell_types : np.ndarray
        Cell type labels (n_cells,).
    gene_names : list of str
        Gene names.
    sender_type : str
        Sender cell type label.
    receiver_type : str
        Receiver cell type label.
    fdr_threshold : float
        FDR threshold for significance.
    min_log2fc : float
        Minimum absolute log2 fold change.

    Returns
    -------
    tuple of (list, list)
        (sender_genes, receiver_genes) - genes upregulated in each type.
    """
    sender_mask = cell_types == sender_type
    receiver_mask = cell_types == receiver_type

    # Test sender vs all others
    other_mask = ~sender_mask
    sender_de = wilcoxon_test(
        expr,
        sender_mask,
        other_mask,
        gene_names,
        fdr_threshold=fdr_threshold,
        min_log2fc=min_log2fc,
    )

    # Get sender-upregulated genes (positive log2FC)
    sender_genes = [
        gene_names[i]
        for i in range(len(gene_names))
        if sender_de.significant_mask[i] and sender_de.log2_fold_change[i] > 0
    ]

    # Test receiver vs all others
    other_mask = ~receiver_mask
    receiver_de = wilcoxon_test(
        expr,
        receiver_mask,
        other_mask,
        gene_names,
        fdr_threshold=fdr_threshold,
        min_log2fc=min_log2fc,
    )

    # Get receiver-upregulated genes (positive log2FC)
    receiver_genes = [
        gene_names[i]
        for i in range(len(gene_names))
        if receiver_de.significant_mask[i] and receiver_de.log2_fold_change[i] > 0
    ]

    return sender_genes, receiver_genes


def rank_genes_by_expression(
    expr: np.ndarray,
    cell_mask: np.ndarray,
    gene_names: list[str],
    top_n: int = 100,
    metric: str = "mean",
) -> list[str]:
    """
    Rank genes by expression in a cell subset.

    Parameters
    ----------
    expr : np.ndarray
        Expression matrix (n_genes, n_cells).
    cell_mask : np.ndarray
        Boolean mask for cells to consider.
    gene_names : list of str
        Gene names.
    top_n : int
        Number of top genes to return.
    metric : str
        Ranking metric ('mean', 'median', 'max', 'fraction').

    Returns
    -------
    list of str
        Top gene names ranked by expression.
    """
    expr_subset = expr[:, cell_mask]

    if metric == "mean":
        scores = np.mean(expr_subset, axis=1)
    elif metric == "median":
        scores = np.median(expr_subset, axis=1)
    elif metric == "max":
        scores = np.max(expr_subset, axis=1)
    elif metric == "fraction":
        scores = np.mean(expr_subset > 0, axis=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    top_indices = np.argsort(scores)[::-1][:top_n]

    return [gene_names[i] for i in top_indices]
