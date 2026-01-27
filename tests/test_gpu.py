"""
GPU vs CPU equivalence tests.

These tests verify that GPU (CuPy) and CPU (NumPy) implementations
produce identical results within numerical tolerance.
"""

import numpy as np
import pytest

from sigdiscovpy.gpu.backend import GPU_AVAILABLE

# Skip all tests if GPU not available
pytestmark = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="CuPy/GPU not available"
)


if GPU_AVAILABLE:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse


class TestNormalizationGPU:
    """GPU vs CPU tests for normalization functions."""

    def test_standardize_vector_equivalence(self):
        """Test standardize_vector produces same result on GPU and CPU."""
        from sigdiscovpy.core.normalization import standardize_vector

        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000).astype(np.float32)

        result_cpu = standardize_vector(x, use_gpu=False)

        x_gpu = cp.asarray(x)
        result_gpu = standardize_vector(x_gpu, use_gpu=True)
        result_gpu_np = cp.asnumpy(result_gpu)

        np.testing.assert_allclose(result_cpu, result_gpu_np, rtol=1e-5)

    def test_standardize_matrix_equivalence(self):
        """Test standardize_matrix produces same result on GPU and CPU."""
        from sigdiscovpy.core.normalization import standardize_matrix

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 500)).astype(np.float32)

        result_cpu = standardize_matrix(X, axis=1, use_gpu=False)

        X_gpu = cp.asarray(X)
        result_gpu = standardize_matrix(X_gpu, axis=1, use_gpu=True)
        result_gpu_np = cp.asnumpy(result_gpu)

        np.testing.assert_allclose(result_cpu, result_gpu_np, rtol=1e-5)


class TestWeightsGPU:
    """GPU vs CPU tests for weight matrix functions."""

    def test_gaussian_weights_equivalence(self):
        """Test Gaussian weights produce same result on GPU and CPU."""
        from sigdiscovpy.core.weights import create_gaussian_weights

        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, (500, 2)).astype(np.float32)

        W_cpu = create_gaussian_weights(coords, radius=100, use_gpu=False)

        coords_gpu = cp.asarray(coords)
        W_gpu = create_gaussian_weights(coords_gpu, radius=100, use_gpu=True)

        # Convert to dense for comparison
        W_cpu_dense = W_cpu.toarray()
        W_gpu_dense = W_gpu.toarray()
        if hasattr(W_gpu_dense, 'get'):
            W_gpu_dense = W_gpu_dense.get()

        np.testing.assert_allclose(W_cpu_dense, W_gpu_dense, rtol=1e-5)

    def test_ring_weights_equivalence(self):
        """Test ring weights produce same result on GPU and CPU."""
        from sigdiscovpy.core.weights import create_ring_weights

        rng = np.random.default_rng(42)
        coords = rng.uniform(0, 1000, (300, 2)).astype(np.float32)

        W_cpu = create_ring_weights(
            coords, outer_radius=150, inner_radius=50, use_gpu=False
        )

        coords_gpu = cp.asarray(coords)
        W_gpu = create_ring_weights(
            coords_gpu, outer_radius=150, inner_radius=50, use_gpu=True
        )

        W_cpu_dense = W_cpu.toarray()
        W_gpu_dense = W_gpu.toarray()
        if hasattr(W_gpu_dense, 'get'):
            W_gpu_dense = W_gpu_dense.get()

        np.testing.assert_allclose(W_cpu_dense, W_gpu_dense, rtol=1e-5)


class TestSpatialLagGPU:
    """GPU vs CPU tests for spatial lag computation."""

    def test_spatial_lag_equivalence(self):
        """Test spatial lag produces same result on GPU and CPU."""
        from sigdiscovpy.core.weights import create_gaussian_weights
        from sigdiscovpy.core.spatial_lag import compute_spatial_lag

        rng = np.random.default_rng(42)
        n = 500
        coords = rng.uniform(0, 1000, (n, 2)).astype(np.float32)
        z = rng.standard_normal(n).astype(np.float32)

        W_cpu = create_gaussian_weights(coords, radius=100, use_gpu=False)
        lag_cpu = compute_spatial_lag(W_cpu, z, use_gpu=False)

        coords_gpu = cp.asarray(coords)
        z_gpu = cp.asarray(z)
        W_gpu = create_gaussian_weights(coords_gpu, radius=100, use_gpu=True)
        lag_gpu = compute_spatial_lag(W_gpu, z_gpu, use_gpu=True)
        lag_gpu_np = cp.asnumpy(lag_gpu)

        np.testing.assert_allclose(lag_cpu, lag_gpu_np, rtol=1e-5)


class TestMetricsGPU:
    """GPU vs CPU tests for metric computations."""

    def test_moran_from_lag_equivalence(self):
        """Test Moran's I from lag produces same result on GPU and CPU."""
        from sigdiscovpy.core.metrics import compute_moran_from_lag

        rng = np.random.default_rng(42)
        n = 500
        z_f = rng.standard_normal(n).astype(np.float32)
        lag_g = rng.standard_normal(n).astype(np.float32)

        result_cpu = compute_moran_from_lag(z_f, lag_g, use_gpu=False)

        z_f_gpu = cp.asarray(z_f)
        lag_g_gpu = cp.asarray(lag_g)
        result_gpu = compute_moran_from_lag(z_f_gpu, lag_g_gpu, use_gpu=True)
        if hasattr(result_gpu, 'get'):
            result_gpu = result_gpu.get()

        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-5)

    def test_ind_from_lag_equivalence(self):
        """Test I_ND from lag produces same result on GPU and CPU."""
        from sigdiscovpy.core.metrics import compute_ind_from_lag

        rng = np.random.default_rng(42)
        n = 500
        z_f = rng.standard_normal(n).astype(np.float32)
        lag_g = rng.standard_normal(n).astype(np.float32)

        result_cpu = compute_ind_from_lag(z_f, lag_g, use_gpu=False)

        z_f_gpu = cp.asarray(z_f)
        lag_g_gpu = cp.asarray(lag_g)
        result_gpu = compute_ind_from_lag(z_f_gpu, lag_g_gpu, use_gpu=True)
        if hasattr(result_gpu, 'get'):
            result_gpu = result_gpu.get()

        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-5)


class TestPairwiseMoranGPU:
    """GPU vs CPU tests for pairwise Moran computation."""

    def test_pairwise_moran_equivalence(self):
        """Test pairwise Moran produces same result on GPU and CPU."""
        from sigdiscovpy.analysis.pairwise_moran import pairwise_moran

        rng = np.random.default_rng(42)
        n_genes = 50
        n_cells = 200

        expr = rng.lognormal(0, 1, (n_genes, n_cells)).astype(np.float32)
        coords = rng.uniform(0, 500, (n_cells, 2)).astype(np.float32)

        result_cpu = pairwise_moran(expr, coords, radius=100, use_gpu=False)
        result_gpu = pairwise_moran(expr, coords, radius=100, use_gpu=True)

        np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-4)


class TestNumericalStabilityGPU:
    """Tests for numerical stability on GPU."""

    def test_small_values(self):
        """Test GPU handles small values correctly."""
        from sigdiscovpy.core.normalization import standardize_vector

        x = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10], dtype=np.float32)

        result_cpu = standardize_vector(x, use_gpu=False)

        x_gpu = cp.asarray(x)
        result_gpu = standardize_vector(x_gpu, use_gpu=True)
        result_gpu_np = cp.asnumpy(result_gpu)

        np.testing.assert_allclose(result_cpu, result_gpu_np, rtol=1e-4)

    def test_large_values(self):
        """Test GPU handles large values correctly."""
        from sigdiscovpy.core.normalization import standardize_vector

        x = np.array([1e8, 2e8, 3e8, 4e8, 5e8], dtype=np.float32)

        result_cpu = standardize_vector(x, use_gpu=False)

        x_gpu = cp.asarray(x)
        result_gpu = standardize_vector(x_gpu, use_gpu=True)
        result_gpu_np = cp.asnumpy(result_gpu)

        np.testing.assert_allclose(result_cpu, result_gpu_np, rtol=1e-4)

    def test_mixed_precision(self):
        """Test GPU handles mixed precision data."""
        from sigdiscovpy.core.metrics import compute_ind_from_lag

        rng = np.random.default_rng(42)

        # Float32
        z_f_32 = rng.standard_normal(100).astype(np.float32)
        lag_g_32 = rng.standard_normal(100).astype(np.float32)

        result_cpu_32 = compute_ind_from_lag(z_f_32, lag_g_32, use_gpu=False)
        result_gpu_32 = compute_ind_from_lag(
            cp.asarray(z_f_32), cp.asarray(lag_g_32), use_gpu=True
        )

        # Float64
        z_f_64 = z_f_32.astype(np.float64)
        lag_g_64 = lag_g_32.astype(np.float64)

        result_cpu_64 = compute_ind_from_lag(z_f_64, lag_g_64, use_gpu=False)
        result_gpu_64 = compute_ind_from_lag(
            cp.asarray(z_f_64), cp.asarray(lag_g_64), use_gpu=True
        )

        # Results should be similar across precisions
        np.testing.assert_allclose(result_cpu_32, float(result_gpu_32), rtol=1e-4)
        np.testing.assert_allclose(result_cpu_64, float(result_gpu_64), rtol=1e-10)


class TestGPUMemory:
    """Tests for GPU memory management."""

    def test_memory_cleanup(self):
        """Test that GPU memory is properly cleaned up."""
        from sigdiscovpy.core.weights import create_gaussian_weights

        initial_mem = cp.get_default_memory_pool().used_bytes()

        for _ in range(5):
            coords = cp.random.uniform(0, 1000, (1000, 2))
            W = create_gaussian_weights(coords, radius=100, use_gpu=True)
            del W, coords
            cp.get_default_memory_pool().free_all_blocks()

        final_mem = cp.get_default_memory_pool().used_bytes()

        # Memory should not grow significantly
        assert final_mem <= initial_mem + 1e7  # Allow 10MB overhead

    def test_large_matrix_chunked(self):
        """Test that large matrices are handled via chunking."""
        from sigdiscovpy.core.weights import create_gaussian_weights

        # Large coordinate array
        coords = cp.random.uniform(0, 1000, (5000, 2)).astype(cp.float32)

        # Should complete without OOM
        W = create_gaussian_weights(coords, radius=100, use_gpu=True)

        assert W.shape == (5000, 5000)
