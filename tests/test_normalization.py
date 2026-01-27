"""Tests for normalization functions."""

import numpy as np

from sigdiscovpy.core.normalization import standardize_matrix, standardize_vector


class TestStandardizeVector:
    """Tests for standardize_vector."""

    def test_basic(self):
        """Test basic standardization."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = standardize_vector(x)

        # Mean should be 0
        assert np.abs(z.mean()) < 1e-10

        # Std should be 1 (population std)
        n = len(z)
        std = np.sqrt(np.sum((z - z.mean()) ** 2) / n)
        assert np.abs(std - 1.0) < 1e-10

    def test_constant_vector(self):
        """Test constant vector returns zeros."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        z = standardize_vector(x)
        assert np.allclose(z, 0.0)

    def test_zero_vector(self):
        """Test zero vector returns zeros."""
        x = np.zeros(10)
        z = standardize_vector(x)
        assert np.allclose(z, 0.0)

    def test_population_std(self):
        """Test that population std (N, not N-1) is used."""
        x = np.array([1.0, 2.0])
        z = standardize_vector(x)

        # Population std of [1, 2] is 0.5
        # (x - mean) / std = (x - 1.5) / 0.5
        expected = np.array([-1.0, 1.0])
        assert np.allclose(z, expected)


class TestStandardizeMatrix:
    """Tests for standardize_matrix."""

    def test_row_wise(self):
        """Test row-wise standardization (genes x cells)."""
        np.random.seed(42)
        X = np.random.randn(100, 1000) * 10 + 5  # 100 genes x 1000 cells

        Z = standardize_matrix(X, axis=1)

        # Each row should have mean ~ 0
        assert np.allclose(Z.mean(axis=1), 0, atol=1e-10)

        # Each row should have population std ~ 1
        for i in range(Z.shape[0]):
            n = Z.shape[1]
            std = np.sqrt(np.sum((Z[i] - Z[i].mean()) ** 2) / n)
            assert np.abs(std - 1.0) < 1e-10 or std == 0  # 0 for constant rows

    def test_column_wise(self):
        """Test column-wise standardization (cells x genes)."""
        np.random.seed(42)
        X = np.random.randn(1000, 100) * 10 + 5  # 1000 cells x 100 genes

        Z = standardize_matrix(X, axis=0)

        # Each column should have mean ~ 0
        assert np.allclose(Z.mean(axis=0), 0, atol=1e-10)

    def test_constant_row(self):
        """Test handling of constant rows."""
        X = np.random.randn(10, 100)
        X[5, :] = 5.0  # Make row 5 constant

        Z = standardize_matrix(X, axis=1)

        # Constant row should be zeros
        assert np.allclose(Z[5, :], 0.0)

        # Other rows should still be standardized
        assert np.allclose(Z[0, :].mean(), 0, atol=1e-10)

    def test_preserves_shape(self):
        """Test output shape matches input."""
        X = np.random.randn(50, 200)
        Z = standardize_matrix(X)
        assert Z.shape == X.shape

    def test_global_normalization(self):
        """Test that standardization uses global mean/std."""
        # This is CRITICAL for correct spatial analysis
        X = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],  # Gene 1
                [10.0, 20.0, 30.0, 40.0],  # Gene 2 (10x Gene 1)
            ]
        )

        Z = standardize_matrix(X, axis=1)

        # Both rows should have same z-scores (just different scale)
        assert np.allclose(Z[0, :], Z[1, :])
