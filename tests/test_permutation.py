"""
Tests for permutation testing module.
"""

import numpy as np

from sigdiscovpy.stats.permutation import (
    batch_permutation_test,
    permutation_test,
)


class TestPermutationTest:
    """Tests for single permutation test."""

    def test_basic_permutation(self):
        """Test basic permutation test functionality."""
        rng = np.random.default_rng(42)
        n = 100

        z_f = rng.standard_normal(n)
        lag_g = rng.standard_normal(n)

        result = permutation_test(
            z_f, lag_g, n_permutations=100, metric="ind", random_seed=42, use_gpu=False
        )

        assert "observed" in result
        assert "pvalue" in result
        assert "null_mean" in result
        assert "null_std" in result
        assert "z_score" in result
        assert 0 <= result["pvalue"] <= 1

    def test_significant_correlation(self):
        """Test that highly correlated data yields low p-value."""
        rng = np.random.default_rng(42)
        n = 100

        # Create correlated vectors
        z_f = rng.standard_normal(n)
        lag_g = z_f * 2 + rng.standard_normal(n) * 0.1

        result = permutation_test(
            z_f,
            lag_g,
            n_permutations=499,
            metric="ind",
            alternative="greater",
            random_seed=42,
            use_gpu=False,
        )

        # Strong correlation should have low p-value
        assert result["pvalue"] < 0.05
        assert result["observed"] > 0.9  # High I_ND

    def test_no_correlation(self):
        """Test that uncorrelated data yields high p-value."""
        rng = np.random.default_rng(42)
        n = 100

        z_f = rng.standard_normal(n)
        lag_g = rng.standard_normal(n)

        result = permutation_test(
            z_f, lag_g, n_permutations=499, metric="ind", random_seed=42, use_gpu=False
        )

        # No correlation should have high p-value (typically > 0.05)
        assert result["pvalue"] > 0.01

    def test_moran_metric(self):
        """Test Moran's I metric."""
        rng = np.random.default_rng(42)
        n = 100

        z_f = rng.standard_normal(n)
        lag_g = z_f + rng.standard_normal(n) * 0.5

        result = permutation_test(
            z_f, lag_g, n_permutations=99, metric="moran", random_seed=42, use_gpu=False
        )

        assert "observed" in result
        assert result["pvalue"] < 0.1  # Should detect correlation

    def test_alternative_greater(self):
        """Test one-sided 'greater' alternative."""
        rng = np.random.default_rng(42)
        n = 100

        z_f = rng.standard_normal(n)
        lag_g = z_f + rng.standard_normal(n) * 0.3

        result = permutation_test(
            z_f, lag_g, n_permutations=199, alternative="greater", random_seed=42, use_gpu=False
        )

        assert result["pvalue"] < 0.05

    def test_alternative_less(self):
        """Test one-sided 'less' alternative."""
        rng = np.random.default_rng(42)
        n = 100

        z_f = rng.standard_normal(n)
        lag_g = -z_f + rng.standard_normal(n) * 0.3  # Negative correlation

        result = permutation_test(
            z_f, lag_g, n_permutations=199, alternative="less", random_seed=42, use_gpu=False
        )

        assert result["pvalue"] < 0.05
        assert result["observed"] < 0

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        rng = np.random.default_rng(42)
        z_f = rng.standard_normal(50)
        lag_g = rng.standard_normal(50)

        result1 = permutation_test(z_f, lag_g, n_permutations=100, random_seed=123, use_gpu=False)

        result2 = permutation_test(z_f, lag_g, n_permutations=100, random_seed=123, use_gpu=False)

        assert result1["pvalue"] == result2["pvalue"]
        assert result1["observed"] == result2["observed"]


class TestBatchPermutationTest:
    """Tests for batch permutation testing."""

    def test_batch_basic(self):
        """Test basic batch permutation test."""
        rng = np.random.default_rng(42)
        n_samples = 50
        n_genes = 10

        z_f = rng.standard_normal(n_samples)
        lag_G = rng.standard_normal((n_samples, n_genes))

        observed, pvalues = batch_permutation_test(
            z_f, lag_G, n_permutations=99, random_seed=42, use_gpu=False
        )

        assert observed.shape == (n_genes,)
        assert pvalues.shape == (n_genes,)
        assert np.all((pvalues >= 0) & (pvalues <= 1))

    def test_batch_identifies_correlated(self):
        """Test that batch test identifies truly correlated pairs."""
        rng = np.random.default_rng(42)
        n_samples = 100
        n_genes = 5

        z_f = rng.standard_normal(n_samples)

        # Create lag matrix: first column correlated, rest random
        lag_G = rng.standard_normal((n_samples, n_genes))
        lag_G[:, 0] = z_f * 2 + rng.standard_normal(n_samples) * 0.1  # Strongly correlated

        observed, pvalues = batch_permutation_test(
            z_f, lag_G, n_permutations=499, random_seed=42, use_gpu=False
        )

        # First gene should have lowest p-value
        assert pvalues[0] < 0.05
        assert pvalues[0] < np.min(pvalues[1:])  # Lowest among all

    def test_batch_moran_metric(self):
        """Test batch test with Moran metric."""
        rng = np.random.default_rng(42)
        n_samples = 50
        n_genes = 5

        z_f = rng.standard_normal(n_samples)
        lag_G = rng.standard_normal((n_samples, n_genes))

        observed, pvalues = batch_permutation_test(
            z_f, lag_G, n_permutations=99, metric="moran", random_seed=42, use_gpu=False
        )

        assert observed.shape == (n_genes,)
        assert pvalues.shape == (n_genes,)


class TestEdgeCases:
    """Tests for edge cases in permutation testing."""

    def test_single_permutation(self):
        """Test with minimal number of permutations."""
        rng = np.random.default_rng(42)
        z_f = rng.standard_normal(20)
        lag_g = rng.standard_normal(20)

        result = permutation_test(z_f, lag_g, n_permutations=1, random_seed=42, use_gpu=False)

        assert 0 <= result["pvalue"] <= 1

    def test_small_sample(self):
        """Test with small sample size."""
        rng = np.random.default_rng(42)
        z_f = rng.standard_normal(5)
        lag_g = rng.standard_normal(5)

        result = permutation_test(z_f, lag_g, n_permutations=50, random_seed=42, use_gpu=False)

        assert "pvalue" in result
        assert 0 <= result["pvalue"] <= 1

    def test_zero_variance_handling(self):
        """Test handling of zero variance input."""
        z_f = np.ones(20)  # Constant
        lag_g = np.random.randn(20)

        result = permutation_test(z_f, lag_g, n_permutations=50, random_seed=42, use_gpu=False)

        # Should handle gracefully (may return NaN)
        assert "pvalue" in result

    def test_nan_in_observed(self):
        """Test handling when observed statistic is NaN."""
        z_f = np.zeros(20)
        lag_g = np.zeros(20)

        result = permutation_test(z_f, lag_g, n_permutations=50, random_seed=42, use_gpu=False)

        # Should return NaN values gracefully
        assert np.isnan(result["observed"])
        assert np.isnan(result["pvalue"])


class TestPValueBounds:
    """Tests for p-value boundary conditions."""

    def test_pvalue_minimum(self):
        """Test that p-value has minimum of 1/(n_permutations+1)."""
        rng = np.random.default_rng(42)
        n = 100

        # Create perfectly correlated data
        z_f = rng.standard_normal(n)
        lag_g = z_f.copy()

        result = permutation_test(z_f, lag_g, n_permutations=99, random_seed=42, use_gpu=False)

        # Minimum p-value should be 1/100 = 0.01
        assert result["pvalue"] >= 1 / 100

    def test_pvalue_maximum(self):
        """Test that p-value is at most 1."""
        rng = np.random.default_rng(42)
        z_f = rng.standard_normal(50)
        lag_g = rng.standard_normal(50)

        result = permutation_test(z_f, lag_g, n_permutations=99, random_seed=42, use_gpu=False)

        assert result["pvalue"] <= 1.0
