"""Tests for core metric functions."""

import numpy as np
import pytest
from sigdiscovpy.core.metrics import (
    compute_moran_from_lag,
    compute_ind_from_lag,
    compute_metric_batch,
)


class TestMoranFromLag:
    """Tests for compute_moran_from_lag."""

    def test_basic(self):
        """Test basic Moran's I computation."""
        z_f = np.array([1.0, -1.0, 1.0, -1.0])
        lag_g = np.array([0.5, -0.5, 0.5, -0.5])
        I = compute_moran_from_lag(z_f, lag_g)
        expected = np.dot(z_f, lag_g) / len(z_f)
        assert np.isclose(I, expected, rtol=1e-10)

    def test_positive_correlation(self):
        """Test positive spatial correlation."""
        z_f = np.array([1.0, 1.0, -1.0, -1.0])
        lag_g = np.array([1.0, 1.0, -1.0, -1.0])
        I = compute_moran_from_lag(z_f, lag_g)
        assert I > 0

    def test_negative_correlation(self):
        """Test negative spatial correlation."""
        z_f = np.array([1.0, -1.0, 1.0, -1.0])
        lag_g = np.array([-1.0, 1.0, -1.0, 1.0])
        I = compute_moran_from_lag(z_f, lag_g)
        assert I < 0

    def test_zero_correlation(self):
        """Test zero spatial correlation."""
        z_f = np.array([1.0, 0.0, -1.0, 0.0])
        lag_g = np.array([0.0, 1.0, 0.0, -1.0])
        I = compute_moran_from_lag(z_f, lag_g)
        assert np.isclose(I, 0.0, atol=1e-10)

    def test_random(self):
        """Test with random data."""
        np.random.seed(42)
        n = 100
        z_f = np.random.randn(n)
        lag_g = np.random.randn(n)
        I = compute_moran_from_lag(z_f, lag_g)
        expected = np.dot(z_f, lag_g) / n
        assert np.isclose(I, expected, rtol=1e-10)


class TestIndFromLag:
    """Tests for compute_ind_from_lag (cosine similarity)."""

    def test_identical_vectors(self):
        """Test I_ND = 1 for identical vectors."""
        z_f = np.array([1.0, 2.0, 3.0, 4.0])
        I_ND = compute_ind_from_lag(z_f, z_f)
        assert np.isclose(I_ND, 1.0, rtol=1e-10)

    def test_opposite_vectors(self):
        """Test I_ND = -1 for opposite vectors."""
        z_f = np.array([1.0, 2.0, 3.0, 4.0])
        lag_g = -z_f
        I_ND = compute_ind_from_lag(z_f, lag_g)
        assert np.isclose(I_ND, -1.0, rtol=1e-10)

    def test_orthogonal_vectors(self):
        """Test I_ND = 0 for orthogonal vectors."""
        z_f = np.array([1.0, 0.0, 0.0, 0.0])
        lag_g = np.array([0.0, 1.0, 0.0, 0.0])
        I_ND = compute_ind_from_lag(z_f, lag_g)
        assert np.isclose(I_ND, 0.0, atol=1e-10)

    def test_zero_vector_returns_nan(self):
        """Test NaN for zero-norm vectors."""
        z_f = np.zeros(4)
        lag_g = np.array([1.0, 2.0, 3.0, 4.0])
        I_ND = compute_ind_from_lag(z_f, lag_g)
        assert np.isnan(I_ND)

    def test_bounded(self):
        """Test I_ND is bounded between -1 and 1."""
        np.random.seed(42)
        for _ in range(10):
            z_f = np.random.randn(100)
            lag_g = np.random.randn(100)
            I_ND = compute_ind_from_lag(z_f, lag_g)
            assert -1 <= I_ND <= 1


class TestMetricBatch:
    """Tests for compute_metric_batch."""

    def test_moran_batch(self):
        """Test batch Moran's I computation."""
        np.random.seed(42)
        n = 100
        n_genes = 10
        z_f = np.random.randn(n)
        lag_G = np.random.randn(n, n_genes)

        result = compute_metric_batch(z_f, lag_G, metric="moran")

        # Verify against individual computation
        for g in range(n_genes):
            expected = compute_moran_from_lag(z_f, lag_G[:, g])
            assert np.isclose(result[g], expected, rtol=1e-10)

    def test_ind_batch(self):
        """Test batch I_ND computation."""
        np.random.seed(42)
        n = 100
        n_genes = 10
        z_f = np.random.randn(n)
        lag_G = np.random.randn(n, n_genes)

        result = compute_metric_batch(z_f, lag_G, metric="ind")

        # Verify against individual computation
        for g in range(n_genes):
            expected = compute_ind_from_lag(z_f, lag_G[:, g])
            assert np.isclose(result[g], expected, rtol=1e-10)

    def test_zero_factor(self):
        """Test batch with zero-norm factor."""
        z_f = np.zeros(100)
        lag_G = np.random.randn(100, 10)

        result = compute_metric_batch(z_f, lag_G, metric="ind")
        assert np.all(np.isnan(result))

    def test_invalid_metric(self):
        """Test invalid metric raises error."""
        z_f = np.random.randn(100)
        lag_G = np.random.randn(100, 10)

        with pytest.raises(ValueError, match="Unknown metric"):
            compute_metric_batch(z_f, lag_G, metric="invalid")
