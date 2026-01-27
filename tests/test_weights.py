"""Tests for weight matrix creation."""

import numpy as np
from scipy import sparse

from sigdiscovpy.core.weights import (
    create_directional_weights,
    create_gaussian_weights,
    create_ring_weights,
    row_normalize_weights,
)


class TestGaussianWeights:
    """Tests for create_gaussian_weights."""

    def test_basic(self):
        """Test basic weight matrix creation."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
        W = create_gaussian_weights(coords, radius=5.0)

        assert W.shape == (4, 4)
        assert sparse.issparse(W)

    def test_sparse_vs_dense(self):
        """Test sparse and dense produce same results."""
        np.random.seed(42)
        coords = np.random.randn(50, 2) * 10

        W_sparse = create_gaussian_weights(coords, radius=5.0, sparse=True)
        W_dense = create_gaussian_weights(coords, radius=5.0, sparse=False)

        assert np.allclose(W_sparse.toarray(), W_dense)

    def test_gaussian_decay(self):
        """Test weights follow Gaussian decay."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        sigma = 10.0 / 3.0
        W = create_gaussian_weights(coords, radius=10.0, sigma=sigma, sparse=False)

        # Weight at distance 1 should be exp(-1^2 / (2*sigma^2))
        expected_1 = np.exp(-1.0 / (2 * sigma**2))
        expected_2 = np.exp(-4.0 / (2 * sigma**2))

        assert np.isclose(W[0, 1], expected_1, rtol=1e-5)
        assert np.isclose(W[0, 2], expected_2, rtol=1e-5)

    def test_no_self_connections(self):
        """Test diagonal is zero when same_spot=False."""
        coords = np.random.randn(10, 2)
        W = create_gaussian_weights(coords, radius=10.0, same_spot=False)

        diag = W.diagonal()
        assert np.allclose(diag, 0.0)

    def test_with_self_connections(self):
        """Test diagonal is 1 when same_spot=True."""
        coords = np.array([[0.0, 0.0], [10.0, 0.0]])
        W = create_gaussian_weights(coords, radius=5.0, same_spot=True)

        # Distance 0 -> weight = exp(0) = 1
        diag = W.diagonal()
        assert np.allclose(diag, 1.0)

    def test_beyond_radius(self):
        """Test zero weights beyond radius."""
        coords = np.array([[0.0, 0.0], [100.0, 0.0]])
        W = create_gaussian_weights(coords, radius=10.0, sparse=False)

        # Points are 100 apart, radius is 10 -> no connection
        assert W[0, 1] == 0
        assert W[1, 0] == 0


class TestRingWeights:
    """Tests for create_ring_weights."""

    def test_basic(self):
        """Test ring weight matrix creation."""
        coords = np.random.randn(50, 2) * 100
        W = create_ring_weights(coords, outer_radius=50, inner_radius=20)

        assert W.shape == (50, 50)

    def test_inner_excluded(self):
        """Test inner radius points are excluded."""
        coords = np.array([[0.0, 0.0], [5.0, 0.0], [30.0, 0.0]])
        W = create_ring_weights(coords, outer_radius=50, inner_radius=10, sparse=False)

        # Point at distance 5 should be excluded (< inner_radius)
        assert W[0, 1] == 0

        # Point at distance 30 should be included
        assert W[0, 2] > 0


class TestRowNormalize:
    """Tests for row_normalize_weights."""

    def test_row_sums(self):
        """Test row sums are 1 after normalization."""
        W = sparse.random(100, 100, density=0.1, format="csr")
        W_norm = row_normalize_weights(W)

        row_sums = np.array(W_norm.sum(axis=1)).ravel()

        # Non-zero rows should sum to 1
        nonzero_rows = np.array(W.sum(axis=1)).ravel() > 0
        assert np.allclose(row_sums[nonzero_rows], 1.0, atol=1e-10)

    def test_zero_row(self):
        """Test zero rows remain zero."""
        data = np.array([1.0, 2.0])
        rows = np.array([0, 0])
        cols = np.array([1, 2])
        W = sparse.csr_matrix((data, (rows, cols)), shape=(3, 3))

        W_norm = row_normalize_weights(W)

        # Rows 1 and 2 should be zeros
        assert np.allclose(W_norm.getrow(1).toarray(), 0)
        assert np.allclose(W_norm.getrow(2).toarray(), 0)


class TestDirectionalWeights:
    """Tests for create_directional_weights."""

    def test_asymmetric(self):
        """Test directional weights are asymmetric."""
        sender_coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        receiver_coords = np.array([[0.5, 0.0], [10.0, 10.0]])

        W = create_directional_weights(sender_coords, receiver_coords, radius=5.0)

        assert W.shape == (2, 2)
        # Sender 0 -> Receiver 0 should have connection
        assert W[0, 0] > 0
        # Sender 0 -> Receiver 1 should not (too far)
        assert W[0, 1] == 0
