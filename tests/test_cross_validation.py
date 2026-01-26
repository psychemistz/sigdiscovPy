"""
Cross-validation tests between sigdiscovPy and sigdiscov R package.

These tests verify numerical equivalence with the R implementation.
Tolerance: rtol=1e-10 for floating point comparison.
"""

import numpy as np
import pytest
import subprocess
import tempfile
from pathlib import Path

# Skip these tests if R is not available
R_AVAILABLE = False
try:
    result = subprocess.run(["Rscript", "--version"], capture_output=True, timeout=5)
    R_AVAILABLE = result.returncode == 0
except Exception:
    pass


def run_r_code(code: str) -> str:
    """Execute R code and return stdout."""
    result = subprocess.run(
        ["Rscript", "-e", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R code failed: {result.stderr}")
    return result.stdout


@pytest.mark.skipif(not R_AVAILABLE, reason="R not available")
class TestCrossValidation:
    """Cross-validation tests against R implementation."""

    def test_moran_from_lag(self):
        """Test compute_moran_from_lag matches R."""
        from sigdiscovpy.core.metrics import compute_moran_from_lag

        np.random.seed(42)
        z_f = np.random.randn(100)
        lag_g = np.random.randn(100)

        # Python result
        py_result = compute_moran_from_lag(z_f, lag_g, use_gpu=False)

        # R result
        z_f_str = ",".join(map(str, z_f))
        lag_g_str = ",".join(map(str, lag_g))
        r_code = f"""
        library(sigdiscov)
        z_f <- c({z_f_str})
        lag_g <- c({lag_g_str})
        cat(compute_moran_from_lag(z_f, lag_g))
        """
        r_result = float(run_r_code(r_code))

        assert np.isclose(py_result, r_result, rtol=1e-10)

    def test_ind_from_lag(self):
        """Test compute_ind_from_lag matches R."""
        from sigdiscovpy.core.metrics import compute_ind_from_lag

        np.random.seed(42)
        z_f = np.random.randn(100)
        lag_g = np.random.randn(100)

        # Python result
        py_result = compute_ind_from_lag(z_f, lag_g, use_gpu=False)

        # R result
        z_f_str = ",".join(map(str, z_f))
        lag_g_str = ",".join(map(str, lag_g))
        r_code = f"""
        library(sigdiscov)
        z_f <- c({z_f_str})
        lag_g <- c({lag_g_str})
        cat(compute_ind_from_lag(z_f, lag_g))
        """
        r_result = float(run_r_code(r_code))

        assert np.isclose(py_result, r_result, rtol=1e-10)

    def test_standardize_vector(self):
        """Test standardize_vector matches R."""
        from sigdiscovpy.core.normalization import standardize_vector

        np.random.seed(42)
        x = np.random.randn(50) * 10 + 5

        # Python result
        py_result = standardize_vector(x, use_gpu=False)

        # R result (manual implementation to avoid dependency)
        r_code = f"""
        x <- c({",".join(map(str, x))})
        n <- length(x)
        m <- mean(x)
        s <- sqrt(sum((x - m)^2) / n)
        z <- (x - m) / s
        cat(paste(z, collapse=","))
        """
        r_output = run_r_code(r_code)
        r_result = np.array([float(v) for v in r_output.strip().split(",")])

        assert np.allclose(py_result, r_result, rtol=1e-10)


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_very_small_values(self):
        """Test handling of very small expression values."""
        from sigdiscovpy.core.metrics import compute_ind_from_lag

        z_f = np.array([1e-15, 2e-15, 3e-15])
        lag_g = np.array([1e-15, 2e-15, 3e-15])

        result = compute_ind_from_lag(z_f, lag_g)
        # Should return NaN due to near-zero norms
        assert np.isnan(result)

    def test_very_large_values(self):
        """Test handling of very large values."""
        from sigdiscovpy.core.metrics import compute_ind_from_lag

        z_f = np.array([1e15, 2e15, 3e15])
        lag_g = np.array([1e15, 2e15, 3e15])

        result = compute_ind_from_lag(z_f, lag_g)
        # Should be 1 (identical vectors)
        assert np.isclose(result, 1.0, rtol=1e-10)

    def test_mixed_magnitude(self):
        """Test handling of mixed magnitude values."""
        from sigdiscovpy.core.normalization import standardize_vector

        x = np.array([1e-10, 1.0, 1e10])
        z = standardize_vector(x)

        # Should still have mean 0, std 1
        assert np.abs(z.mean()) < 1e-10
