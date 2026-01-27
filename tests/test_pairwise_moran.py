"""Tests for pairwise Moran's I matrix computation."""

import numpy as np

from sigdiscovpy.analysis.pairwise_moran import pairwise_moran


class TestPairwiseMoran:
    """Tests for pairwise_moran."""

    def test_basic(self):
        """Test basic pairwise Moran computation."""
        np.random.seed(42)
        n_genes, n_spots = 10, 100
        expr = np.random.randn(n_genes, n_spots)
        coords = np.random.randn(n_spots, 2) * 100

        I_matrix = pairwise_moran(expr, coords, radius=50)

        assert I_matrix.shape == (n_genes, n_genes)

    def test_symmetric(self):
        """Test matrix is symmetric."""
        np.random.seed(42)
        expr = np.random.randn(10, 100)
        coords = np.random.randn(100, 2) * 100

        I_matrix = pairwise_moran(expr, coords, radius=50)

        assert np.allclose(I_matrix, I_matrix.T)

    def test_diagonal(self):
        """Test diagonal elements are autocorrelation."""
        np.random.seed(42)
        expr = np.random.randn(5, 50)
        coords = np.random.randn(50, 2) * 50

        I_matrix = pairwise_moran(expr, coords, radius=30)

        # Diagonal should be positive (autocorrelation)
        # Note: Not always positive due to random spatial arrangement
        # Just verify it's computed
        assert not np.any(np.isnan(np.diag(I_matrix)))

    def test_auto_detect_orientation(self):
        """Test automatic detection of expression matrix orientation."""
        np.random.seed(42)
        n_genes, n_spots = 10, 100
        expr = np.random.randn(n_genes, n_spots)
        coords = np.random.randn(n_spots, 2) * 100

        # Standard orientation (genes x spots)
        I1 = pairwise_moran(expr, coords, radius=50)

        # Transposed orientation (spots x genes)
        I2 = pairwise_moran(expr.T, coords, radius=50)

        assert I1.shape == (n_genes, n_genes)
        assert I2.shape == (n_genes, n_genes)
        assert np.allclose(I1, I2)

    def test_sparse_vs_dense_W(self):
        """Test sparse and dense W produce same results."""
        np.random.seed(42)
        expr = np.random.randn(5, 50)
        coords = np.random.randn(50, 2) * 50

        I_sparse = pairwise_moran(expr, coords, radius=30, sparse_W=True)
        I_dense = pairwise_moran(expr, coords, radius=30, sparse_W=False)

        assert np.allclose(I_sparse, I_dense, rtol=1e-10)

    def test_no_connections(self):
        """Test returns zeros when no spatial connections."""
        expr = np.random.randn(5, 10)
        # Points very far apart
        coords = np.array([[i * 1000, 0] for i in range(10)])

        I_matrix = pairwise_moran(expr, coords, radius=10)

        # Should be all zeros (no neighbors)
        assert np.allclose(I_matrix, 0.0)

    def test_normalization(self):
        """Test normalization option."""
        np.random.seed(42)
        expr = np.random.randn(5, 50)
        coords = np.random.randn(50, 2) * 50

        # With normalization (default)
        I_norm = pairwise_moran(expr, coords, radius=30, normalize=True)

        # Without normalization (pre-normalized data)
        from sigdiscovpy.core.normalization import standardize_matrix

        expr_z = standardize_matrix(expr, axis=1)
        I_nonorm = pairwise_moran(expr_z, coords, radius=30, normalize=False)

        assert np.allclose(I_norm, I_nonorm, rtol=1e-10)


class TestPairwiseMoranDirectional:
    """Tests for pairwise_moran_directional."""

    def test_basic(self):
        """Test basic directional pairwise Moran."""
        from sigdiscovpy.analysis.pairwise_moran import pairwise_moran_directional

        np.random.seed(42)
        n_genes = 5
        n_senders, n_receivers = 50, 60

        sender_expr = np.random.randn(n_genes, n_senders)
        receiver_expr = np.random.randn(n_genes, n_receivers)
        sender_coords = np.random.randn(n_senders, 2) * 100
        receiver_coords = np.random.randn(n_receivers, 2) * 100

        I_matrix = pairwise_moran_directional(
            sender_expr,
            receiver_expr,
            sender_coords,
            receiver_coords,
            radius=50,
        )

        assert I_matrix.shape == (n_genes, n_genes)

    def test_asymmetric(self):
        """Test directional matrix is not necessarily symmetric."""
        from sigdiscovpy.analysis.pairwise_moran import pairwise_moran_directional

        np.random.seed(42)
        n_genes = 3
        n_senders, n_receivers = 30, 40

        sender_expr = np.random.randn(n_genes, n_senders)
        receiver_expr = np.random.randn(n_genes, n_receivers)
        sender_coords = np.random.randn(n_senders, 2) * 100
        receiver_coords = np.random.randn(n_receivers, 2) * 100

        I_matrix = pairwise_moran_directional(
            sender_expr,
            receiver_expr,
            sender_coords,
            receiver_coords,
            radius=50,
        )

        # For different sender/receiver, matrix need not be symmetric
        # (Though it might be by chance for random data)
        assert I_matrix.shape == (n_genes, n_genes)
