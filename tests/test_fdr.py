"""Tests for FDR correction."""

import numpy as np
import pytest
from sigdiscovpy.stats.fdr import apply_fdr_correction


class TestFDRCorrection:
    """Tests for apply_fdr_correction."""

    def test_bonferroni(self):
        """Test Bonferroni correction."""
        pvalues = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = apply_fdr_correction(pvalues, method="bonferroni")

        expected = pvalues * len(pvalues)
        assert np.allclose(result["adjusted_pvalues"], expected)

    def test_bonferroni_capped(self):
        """Test Bonferroni doesn't exceed 1."""
        pvalues = np.array([0.5, 0.8])
        result = apply_fdr_correction(pvalues, method="bonferroni")

        assert np.all(result["adjusted_pvalues"] <= 1.0)

    def test_bh_basic(self):
        """Test BH correction basic case."""
        pvalues = np.array([0.01, 0.04, 0.05])
        result = apply_fdr_correction(pvalues, method="bh")

        # BH adjusted p-values should be >= original
        assert np.all(result["adjusted_pvalues"] >= pvalues)

    def test_bh_monotonicity(self):
        """Test BH maintains monotonicity."""
        pvalues = np.array([0.001, 0.01, 0.02, 0.05, 0.1])
        result = apply_fdr_correction(pvalues, method="bh")

        # Sorted adjusted p-values should be monotonically increasing
        sorted_idx = np.argsort(pvalues)
        sorted_adj = result["adjusted_pvalues"][sorted_idx]
        assert np.all(sorted_adj[1:] >= sorted_adj[:-1])

    def test_by_more_conservative(self):
        """Test BY is more conservative than BH."""
        pvalues = np.array([0.01, 0.02, 0.03])

        bh_result = apply_fdr_correction(pvalues, method="bh")
        by_result = apply_fdr_correction(pvalues, method="by")

        # BY should give higher (more conservative) adjusted p-values
        assert np.all(by_result["adjusted_pvalues"] >= bh_result["adjusted_pvalues"])

    def test_significance_count(self):
        """Test n_significant is correct."""
        pvalues = np.array([0.001, 0.01, 0.1, 0.5])
        result = apply_fdr_correction(pvalues, method="bh", alpha=0.05)

        expected_sig = np.sum(result["adjusted_pvalues"] < 0.05)
        assert result["n_significant"] == expected_sig

    def test_nan_handling(self):
        """Test NaN p-values are handled."""
        pvalues = np.array([0.01, np.nan, 0.03])
        result = apply_fdr_correction(pvalues, method="bh")

        assert np.isnan(result["adjusted_pvalues"][1])
        assert not np.isnan(result["adjusted_pvalues"][0])
        assert not np.isnan(result["adjusted_pvalues"][2])

    def test_empty_input(self):
        """Test empty input."""
        pvalues = np.array([])
        result = apply_fdr_correction(pvalues, method="bh")

        assert len(result["adjusted_pvalues"]) == 0
        assert result["n_significant"] == 0

    def test_invalid_method(self):
        """Test invalid method raises error."""
        pvalues = np.array([0.01, 0.02])

        with pytest.raises(ValueError, match="Unknown method"):
            apply_fdr_correction(pvalues, method="invalid")

    def test_all_significant(self):
        """Test all p-values significant after correction."""
        pvalues = np.array([0.001, 0.001, 0.001])
        result = apply_fdr_correction(pvalues, method="bh", alpha=0.05)

        assert result["n_significant"] == 3

    def test_none_significant(self):
        """Test no p-values significant after correction."""
        pvalues = np.array([0.5, 0.6, 0.7])
        result = apply_fdr_correction(pvalues, method="bh", alpha=0.05)

        assert result["n_significant"] == 0
