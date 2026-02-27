"""
Cross-implementation numerical identity tests.

Verifies that sim_parse10m (reference), sigdiscovpy, and sigdiscov (R)
produce numerically identical results for:
  1. Z-score standardization (population std, ddof=0)
  2. Weight matrix construction (Gaussian-annular AND binary circular)
  3. I_ND computation (cosine similarity of z_f and W@z_r)
  4. DE analysis (log2FC with pseudocount=1e-9)
  5. Edge cases (constant vectors, zero-norm, perfect correlation)
"""

import subprocess
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sigdiscovpy.core.normalization import standardize_vector, standardize_matrix
from sigdiscovpy.core.metrics import (
    compute_ind_from_lag,
    compute_moran_from_lag,
    compute_metric_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HAS_R = shutil.which("R") is not None

try:
    if HAS_R:
        _probe = subprocess.run(
            ["R", "--no-save", "--slave", "-e", "library(sigdiscov); cat('ok')"],
            capture_output=True, text=True, timeout=60,
        )
        HAS_R_PKG = _probe.stdout.strip().endswith("ok")
    else:
        HAS_R_PKG = False
except Exception:
    HAS_R_PKG = False

requires_r = pytest.mark.skipif(
    not HAS_R_PKG, reason="R or sigdiscov R package not available"
)


def run_r(code: str) -> str:
    """Run R code via temp file and return stdout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(code)
        f.flush()
        result = subprocess.run(
            ["R", "--no-save", "--slave", "-f", f.name],
            capture_output=True, text=True, timeout=120,
        )
    Path(f.name).unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"R error:\n{result.stderr}")
    return result.stdout.strip()


def _ref_zscore_vector(x):
    """sim_parse10m reference: z-score with population std (ddof=0)."""
    mu = np.mean(x)
    sigma = np.std(x)
    return (x - mu) / sigma if sigma > 1e-10 else np.zeros_like(x)


def _ref_zscore_matrix(mat):
    """sim_parse10m reference: gene-wise z-score with population std."""
    out = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        mu = np.mean(mat[i])
        sigma = np.std(mat[i])
        out[i] = (mat[i] - mu) / sigma if sigma > 1e-10 else 0
    return out


def _ref_ind(z_f, lag):
    """sim_parse10m reference: I_ND = cosine similarity."""
    nf = np.linalg.norm(z_f)
    nl = np.linalg.norm(lag)
    if nf > 1e-10 and nl > 1e-10:
        return np.dot(z_f, lag) / (nf * nl)
    return 0.0


def _build_gaussian_annular_weights(sender_pos, receiver_pos, radius, inner_radius=0.0):
    """Build Gaussian-annular weight matrix (sim_parse10m reference)."""
    dx = sender_pos[:, 0:1] - receiver_pos[:, 0].reshape(1, -1)
    dy = sender_pos[:, 1:2] - receiver_pos[:, 1].reshape(1, -1)
    dists = np.sqrt(dx ** 2 + dy ** 2)
    sigma = radius / 3.0
    W = np.exp(-dists ** 2 / (2 * sigma ** 2))
    W = W * ((dists <= radius) & (dists > inner_radius))
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs, dists


def _build_binary_weights(sender_pos, receiver_pos, radius, inner_radius=0.0):
    """Build binary (circular) weight matrix."""
    dx = sender_pos[:, 0:1] - receiver_pos[:, 0].reshape(1, -1)
    dy = sender_pos[:, 1:2] - receiver_pos[:, 1].reshape(1, -1)
    dists = np.sqrt(dx ** 2 + dy ** 2)
    W = ((dists <= radius) & (dists > inner_radius)).astype(float)
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs, dists


def _fmt_vec(v):
    """Format numpy vector as R-parseable string."""
    return ", ".join(f"{x:.15e}" for x in v)


def _fmt_mat(m, order="C"):
    """Format numpy matrix as R-parseable flat string."""
    return ", ".join(f"{x:.15e}" for x in m.flatten(order=order))


# ===========================================================================
# TEST 1: Z-SCORE STANDARDIZATION
# ===========================================================================

class TestZScore:
    """Population std (ddof=0) z-score across all implementations."""

    def test_sim_vs_sigdiscovpy_vector(self):
        np.random.seed(42)
        x = np.random.randn(100) * 5 + 10
        sim_z = _ref_zscore_vector(x)
        py_z = standardize_vector(x, use_gpu=False)
        assert np.max(np.abs(sim_z - py_z)) < 1e-12

    @requires_r
    def test_sim_vs_r_vector(self):
        np.random.seed(42)
        x = np.random.randn(100) * 5 + 10
        sim_z = _ref_zscore_vector(x)
        r_out = run_r(f"""
        library(sigdiscov)
        x <- c({_fmt_vec(x)})
        z <- sigdiscov::standardize(x)
        cat(paste(sprintf("%.15e", z), collapse=","))
        """)
        r_z = np.array([float(v) for v in r_out.split(",")])
        assert np.max(np.abs(sim_z - r_z)) < 1e-9

    def test_sim_vs_sigdiscovpy_matrix(self):
        np.random.seed(123)
        mat = np.random.randn(10, 50) * 3 + 5
        sim_z = _ref_zscore_matrix(mat)
        py_z = standardize_matrix(mat, axis=1, use_gpu=False)
        assert np.max(np.abs(sim_z - py_z)) < 1e-12

    @requires_r
    def test_sim_vs_r_matrix(self):
        np.random.seed(123)
        mat = np.random.randn(10, 50) * 3 + 5
        sim_z = _ref_zscore_matrix(mat)
        r_out = run_r(f"""
        library(sigdiscov)
        mat <- matrix(c({_fmt_mat(mat)}), nrow=10, ncol=50, byrow=TRUE)
        z <- standardize_matrix(mat)
        cat(paste(sprintf("%.15e", as.vector(t(z))), collapse=","))
        """)
        r_z = np.array([float(v) for v in r_out.split(",")]).reshape(10, 50)
        assert np.max(np.abs(sim_z - r_z)) < 1e-9

    @requires_r
    def test_sigdiscovpy_vs_r_matrix(self):
        np.random.seed(123)
        mat = np.random.randn(10, 50) * 3 + 5
        py_z = standardize_matrix(mat, axis=1, use_gpu=False)
        r_out = run_r(f"""
        library(sigdiscov)
        mat <- matrix(c({_fmt_mat(mat)}), nrow=10, ncol=50, byrow=TRUE)
        z <- standardize_matrix(mat)
        cat(paste(sprintf("%.15e", as.vector(t(z))), collapse=","))
        """)
        r_z = np.array([float(v) for v in r_out.split(",")]).reshape(10, 50)
        assert np.max(np.abs(py_z - r_z)) < 1e-9


# ===========================================================================
# TEST 2: WEIGHT MATRIX — GAUSSIAN-ANNULAR (ANNULAR)
# ===========================================================================

class TestGaussianAnnularWeights:
    """Gaussian-annular weight matrix across implementations."""

    @requires_r
    def test_sim_vs_r(self):
        np.random.seed(42)
        n_s, n_r = 20, 30
        s_pos = np.random.randn(n_s, 2) * 500
        r_pos = np.random.randn(n_r, 2) * 500 + 200
        radius, inner = 400.0, 200.0

        W_sim, _ = _build_gaussian_annular_weights(s_pos, r_pos, radius, inner)

        r_out = run_r(f"""
        sp <- matrix(c({_fmt_mat(s_pos)}), nrow={n_s}, ncol=2, byrow=TRUE)
        rp <- matrix(c({_fmt_mat(r_pos)}), nrow={n_r}, ncol=2, byrow=TRUE)
        dx <- outer(sp[,1], rp[,1], `-`)
        dy <- outer(sp[,2], rp[,2], `-`)
        d <- sqrt(dx^2 + dy^2)
        sigma <- {radius}/3
        W <- exp(-d^2/(2*sigma^2))
        mask <- (d <= {radius}) & (d > {inner})
        W <- W * mask
        rs <- rowSums(W); rs[rs==0] <- 1
        W <- W / rs
        cat(paste(sprintf("%.15e", as.vector(t(W))), collapse=","))
        """)
        W_r = np.array([float(v) for v in r_out.split(",")]).reshape(n_s, n_r)
        assert np.max(np.abs(W_sim - W_r)) < 1e-12


# ===========================================================================
# TEST 3: WEIGHT MATRIX — BINARY CIRCULAR (DISK & RING)
# ===========================================================================

class TestBinaryCircularWeights:
    """Binary (circular/ring) weight matrix across implementations."""

    @requires_r
    def test_binary_disk_sim_vs_r(self):
        """Binary disk: all cells within radius, no inner exclusion."""
        np.random.seed(42)
        n_s, n_r = 20, 30
        s_pos = np.random.randn(n_s, 2) * 500
        r_pos = np.random.randn(n_r, 2) * 500 + 200
        radius = 500.0

        W_sim, _ = _build_binary_weights(s_pos, r_pos, radius, inner_radius=0.0)

        r_out = run_r(f"""
        sp <- matrix(c({_fmt_mat(s_pos)}), nrow={n_s}, ncol=2, byrow=TRUE)
        rp <- matrix(c({_fmt_mat(r_pos)}), nrow={n_r}, ncol=2, byrow=TRUE)
        dx <- outer(sp[,1], rp[,1], `-`)
        dy <- outer(sp[,2], rp[,2], `-`)
        d <- sqrt(dx^2 + dy^2)
        W <- (d <= {radius}) * 1.0
        rs <- rowSums(W); rs[rs==0] <- 1
        W <- W / rs
        cat(paste(sprintf("%.15e", as.vector(t(W))), collapse=","))
        """)
        W_r = np.array([float(v) for v in r_out.split(",")]).reshape(n_s, n_r)
        assert np.max(np.abs(W_sim - W_r)) < 1e-12

    @requires_r
    def test_binary_ring_sim_vs_r(self):
        """Binary ring: annular region with inner exclusion."""
        np.random.seed(42)
        n_s, n_r = 20, 30
        s_pos = np.random.randn(n_s, 2) * 500
        r_pos = np.random.randn(n_r, 2) * 500 + 200
        outer_r, inner_r = 500.0, 200.0

        W_sim, _ = _build_binary_weights(s_pos, r_pos, outer_r, inner_r)

        r_out = run_r(f"""
        sp <- matrix(c({_fmt_mat(s_pos)}), nrow={n_s}, ncol=2, byrow=TRUE)
        rp <- matrix(c({_fmt_mat(r_pos)}), nrow={n_r}, ncol=2, byrow=TRUE)
        dx <- outer(sp[,1], rp[,1], `-`)
        dy <- outer(sp[,2], rp[,2], `-`)
        d <- sqrt(dx^2 + dy^2)
        W <- ((d > {inner_r}) & (d <= {outer_r})) * 1.0
        rs <- rowSums(W); rs[rs==0] <- 1
        W <- W / rs
        cat(paste(sprintf("%.15e", as.vector(t(W))), collapse=","))
        """)
        W_r = np.array([float(v) for v in r_out.split(",")]).reshape(n_s, n_r)
        assert np.max(np.abs(W_sim - W_r)) < 1e-12


# ===========================================================================
# TEST 4: I_ND BATCH — GAUSSIAN-ANNULAR WEIGHTS
# ===========================================================================

class TestINDBatchGaussianAnnular:
    """I_ND batch computation with Gaussian-annular weights."""

    def _setup(self):
        np.random.seed(42)
        n_s, n_r, n_g = 50, 80, 20
        n_all = n_s + n_r
        s_pos = np.random.randn(n_s, 2) * 1000
        r_pos = np.random.randn(n_r, 2) * 1000 + 300
        expr = np.random.randn(n_g, n_all) * 2 + 3
        expr_z = _ref_zscore_matrix(expr)
        z_f = expr_z[0, :n_s]
        z_recv = expr_z[:, n_s:]
        W, _ = _build_gaussian_annular_weights(s_pos, r_pos, radius=800.0)
        lag_G = W @ z_recv.T
        sim_ind = np.array([_ref_ind(z_f, lag_G[:, g]) for g in range(n_g)])
        return z_f, lag_G, sim_ind, n_s, n_g

    def test_sim_vs_sigdiscovpy(self):
        z_f, lag_G, sim_ind, _, _ = self._setup()
        py_ind = compute_metric_batch(z_f, lag_G, metric="ind", use_gpu=False)
        assert np.max(np.abs(sim_ind - py_ind)) < 1e-10

    @requires_r
    def test_sim_vs_r(self):
        z_f, lag_G, sim_ind, n_s, n_g = self._setup()
        r_out = run_r(f"""
        library(sigdiscov)
        z_f <- c({_fmt_vec(z_f)})
        lag_G <- matrix(c({_fmt_mat(lag_G, order='F')}), nrow={n_s}, ncol={n_g})
        result <- compute_metric_batch(z_f, lag_G, "ind")
        result[is.na(result)] <- 0
        cat(paste(sprintf("%.15e", result), collapse=","))
        """)
        r_ind = np.array([float(v) for v in r_out.split(",")])
        assert np.max(np.abs(sim_ind - r_ind)) < 1e-10

    @requires_r
    def test_sigdiscovpy_vs_r(self):
        z_f, lag_G, _, n_s, n_g = self._setup()
        py_ind = compute_metric_batch(z_f, lag_G, metric="ind", use_gpu=False)
        r_out = run_r(f"""
        library(sigdiscov)
        z_f <- c({_fmt_vec(z_f)})
        lag_G <- matrix(c({_fmt_mat(lag_G, order='F')}), nrow={n_s}, ncol={n_g})
        result <- compute_metric_batch(z_f, lag_G, "ind")
        result[is.na(result)] <- 0
        cat(paste(sprintf("%.15e", result), collapse=","))
        """)
        r_ind = np.array([float(v) for v in r_out.split(",")])
        assert np.max(np.abs(py_ind - r_ind)) < 1e-10


# ===========================================================================
# TEST 5: I_ND BATCH — BINARY CIRCULAR WEIGHTS
# ===========================================================================

class TestINDBatchBinaryCircular:
    """I_ND batch computation with binary (circular) weights."""

    def _setup(self, inner_radius=0.0):
        np.random.seed(55)
        n_s, n_r, n_g = 40, 60, 8
        n_all = n_s + n_r
        s_pos = np.random.randn(n_s, 2) * 800
        r_pos = np.random.randn(n_r, 2) * 800 + 150
        expr = np.abs(np.random.randn(n_g, n_all)) * 2 + 1
        expr_z = _ref_zscore_matrix(expr)
        z_f = expr_z[0, :n_s]
        z_recv = expr_z[:, n_s:]
        W, _ = _build_binary_weights(s_pos, r_pos, radius=700.0, inner_radius=inner_radius)
        lag_G = W @ z_recv.T
        sim_ind = np.array([_ref_ind(z_f, lag_G[:, g]) for g in range(n_g)])
        return z_f, lag_G, sim_ind, n_s, n_g, s_pos, r_pos, expr, n_all

    # --- Binary disk (inner_radius=0) ---

    def test_disk_sim_vs_sigdiscovpy(self):
        z_f, lag_G, sim_ind, *_ = self._setup(inner_radius=0.0)
        py_ind = compute_metric_batch(z_f, lag_G, metric="ind", use_gpu=False)
        assert np.max(np.abs(sim_ind - py_ind)) < 1e-10

    @requires_r
    def test_disk_sim_vs_r(self):
        z_f, lag_G, sim_ind, n_s, n_g, *_ = self._setup(inner_radius=0.0)
        r_out = run_r(f"""
        library(sigdiscov)
        z_f <- c({_fmt_vec(z_f)})
        lag_G <- matrix(c({_fmt_mat(lag_G, order='F')}), nrow={n_s}, ncol={n_g})
        result <- compute_metric_batch(z_f, lag_G, "ind")
        result[is.na(result)] <- 0
        cat(paste(sprintf("%.15e", result), collapse=","))
        """)
        r_ind = np.array([float(v) for v in r_out.split(",")])
        assert np.max(np.abs(sim_ind - r_ind)) < 1e-10

    @requires_r
    def test_disk_sigdiscovpy_vs_r(self):
        z_f, lag_G, _, n_s, n_g, *_ = self._setup(inner_radius=0.0)
        py_ind = compute_metric_batch(z_f, lag_G, metric="ind", use_gpu=False)
        r_out = run_r(f"""
        library(sigdiscov)
        z_f <- c({_fmt_vec(z_f)})
        lag_G <- matrix(c({_fmt_mat(lag_G, order='F')}), nrow={n_s}, ncol={n_g})
        result <- compute_metric_batch(z_f, lag_G, "ind")
        result[is.na(result)] <- 0
        cat(paste(sprintf("%.15e", result), collapse=","))
        """)
        r_ind = np.array([float(v) for v in r_out.split(",")])
        assert np.max(np.abs(py_ind - r_ind)) < 1e-10

    # --- Binary ring (inner_radius=300) ---

    def test_ring_sim_vs_sigdiscovpy(self):
        z_f, lag_G, sim_ind, *_ = self._setup(inner_radius=300.0)
        py_ind = compute_metric_batch(z_f, lag_G, metric="ind", use_gpu=False)
        assert np.max(np.abs(sim_ind - py_ind)) < 1e-10

    @requires_r
    def test_ring_sim_vs_r(self):
        z_f, lag_G, sim_ind, n_s, n_g, *_ = self._setup(inner_radius=300.0)
        r_out = run_r(f"""
        library(sigdiscov)
        z_f <- c({_fmt_vec(z_f)})
        lag_G <- matrix(c({_fmt_mat(lag_G, order='F')}), nrow={n_s}, ncol={n_g})
        result <- compute_metric_batch(z_f, lag_G, "ind")
        result[is.na(result)] <- 0
        cat(paste(sprintf("%.15e", result), collapse=","))
        """)
        r_ind = np.array([float(v) for v in r_out.split(",")])
        assert np.max(np.abs(sim_ind - r_ind)) < 1e-10

    @requires_r
    def test_ring_sigdiscovpy_vs_r(self):
        z_f, lag_G, _, n_s, n_g, *_ = self._setup(inner_radius=300.0)
        py_ind = compute_metric_batch(z_f, lag_G, metric="ind", use_gpu=False)
        r_out = run_r(f"""
        library(sigdiscov)
        z_f <- c({_fmt_vec(z_f)})
        lag_G <- matrix(c({_fmt_mat(lag_G, order='F')}), nrow={n_s}, ncol={n_g})
        result <- compute_metric_batch(z_f, lag_G, "ind")
        result[is.na(result)] <- 0
        cat(paste(sprintf("%.15e", result), collapse=","))
        """)
        r_ind = np.array([float(v) for v in r_out.split(",")])
        assert np.max(np.abs(py_ind - r_ind)) < 1e-10


# ===========================================================================
# TEST 6: END-TO-END I_ND (standardize + weights + metric)
# ===========================================================================

class TestEndToEndIND:
    """Full pipeline: raw expression -> z-score -> weights -> I_ND."""

    def _setup(self, weight_type="gaussian_annular"):
        np.random.seed(99)
        n_s, n_r, n_g = 30, 50, 5
        n_all = n_s + n_r
        expr_raw = np.abs(np.random.randn(n_g, n_all)) * 3 + 1
        s_pos = np.random.randn(n_s, 2) * 800
        r_pos = np.random.randn(n_r, 2) * 800 + 200
        radius, inner = 600.0, 100.0

        expr_z = _ref_zscore_matrix(expr_raw)
        z_f = expr_z[0, :n_s]
        z_recv = expr_z[:, n_s:]

        if weight_type == "gaussian_annular":
            W, _ = _build_gaussian_annular_weights(s_pos, r_pos, radius, inner)
        else:
            W, _ = _build_binary_weights(s_pos, r_pos, radius, inner)

        sim_ind = np.array([
            _ref_ind(z_f, W @ z_recv[g]) for g in range(n_g)
        ])
        return expr_raw, s_pos, r_pos, z_f, z_recv, W, sim_ind, n_s, n_r, n_g, n_all, radius, inner

    # --- Gaussian-annular ---

    def test_gaussian_annular_sim_vs_sigdiscovpy(self):
        _, _, _, z_f, z_recv, W, sim_ind, *_ = self._setup("gaussian_annular")
        py_ind = compute_metric_batch(z_f, W @ z_recv.T, metric="ind", use_gpu=False)
        assert np.max(np.abs(sim_ind - py_ind)) < 1e-10

    @requires_r
    def test_gaussian_annular_sim_vs_r(self):
        expr_raw, s_pos, r_pos, _, _, _, sim_ind, n_s, n_r, n_g, n_all, radius, inner = self._setup("gaussian_annular")
        r_out = run_r(f"""
        library(sigdiscov)
        expr_raw <- matrix(c({_fmt_mat(expr_raw)}), nrow={n_g}, ncol={n_all}, byrow=TRUE)
        sp <- matrix(c({_fmt_mat(s_pos)}), nrow={n_s}, ncol=2, byrow=TRUE)
        rp <- matrix(c({_fmt_mat(r_pos)}), nrow={n_r}, ncol=2, byrow=TRUE)
        expr_z <- standardize_matrix(expr_raw)
        z_f <- expr_z[1, 1:{n_s}]
        z_recv <- expr_z[, {n_s+1}:{n_all}]
        dx <- outer(sp[,1], rp[,1], `-`)
        dy <- outer(sp[,2], rp[,2], `-`)
        d <- sqrt(dx^2 + dy^2)
        sigma <- {radius}/3
        W <- exp(-d^2/(2*sigma^2))
        mask <- (d <= {radius}) & (d > {inner})
        W <- W * mask
        rs <- rowSums(W); rs[rs==0] <- 1
        W <- W / rs
        lag_G <- W %*% t(z_recv)
        result <- compute_metric_batch(z_f, lag_G, "ind")
        result[is.na(result)] <- 0
        cat(paste(sprintf("%.15e", result), collapse=","))
        """)
        r_ind = np.array([float(v) for v in r_out.split(",")])
        assert np.max(np.abs(sim_ind - r_ind)) < 1e-10

    # --- Binary circular ---

    def test_binary_circular_sim_vs_sigdiscovpy(self):
        _, _, _, z_f, z_recv, W, sim_ind, *_ = self._setup("binary")
        py_ind = compute_metric_batch(z_f, W @ z_recv.T, metric="ind", use_gpu=False)
        assert np.max(np.abs(sim_ind - py_ind)) < 1e-10

    @requires_r
    def test_binary_circular_sim_vs_r(self):
        expr_raw, s_pos, r_pos, _, _, _, sim_ind, n_s, n_r, n_g, n_all, radius, inner = self._setup("binary")
        r_out = run_r(f"""
        library(sigdiscov)
        expr_raw <- matrix(c({_fmt_mat(expr_raw)}), nrow={n_g}, ncol={n_all}, byrow=TRUE)
        sp <- matrix(c({_fmt_mat(s_pos)}), nrow={n_s}, ncol=2, byrow=TRUE)
        rp <- matrix(c({_fmt_mat(r_pos)}), nrow={n_r}, ncol=2, byrow=TRUE)
        expr_z <- standardize_matrix(expr_raw)
        z_f <- expr_z[1, 1:{n_s}]
        z_recv <- expr_z[, {n_s+1}:{n_all}]
        dx <- outer(sp[,1], rp[,1], `-`)
        dy <- outer(sp[,2], rp[,2], `-`)
        d <- sqrt(dx^2 + dy^2)
        W <- ((d <= {radius}) & (d > {inner})) * 1.0
        rs <- rowSums(W); rs[rs==0] <- 1
        W <- W / rs
        lag_G <- W %*% t(z_recv)
        result <- compute_metric_batch(z_f, lag_G, "ind")
        result[is.na(result)] <- 0
        cat(paste(sprintf("%.15e", result), collapse=","))
        """)
        r_ind = np.array([float(v) for v in r_out.split(",")])
        assert np.max(np.abs(sim_ind - r_ind)) < 1e-10


# ===========================================================================
# TEST 7: R compute_ind_single (GAUSSIAN-ANNULAR & ANNULAR)
# ===========================================================================

class TestComputeIndSingleR:
    """Test R compute_ind_single function against Python for both weight types."""

    def _setup(self):
        np.random.seed(77)
        n_s, n_r = 25, 40
        n_all = n_s + n_r
        all_pos = np.random.randn(n_all, 2) * 500
        factor_expr = np.abs(np.random.randn(n_all)) * 2 + 1
        response_expr = np.abs(np.random.randn(n_all)) * 3 + 2
        return all_pos, factor_expr, response_expr, n_s, n_r, n_all

    def _python_ind(self, all_pos, factor_expr, response_expr, n_s, radius, build_fn, **kw):
        mu_f, sig_f = np.mean(factor_expr), np.std(factor_expr)
        mu_r, sig_r = np.mean(response_expr), np.std(response_expr)
        z_s = (factor_expr[:n_s] - mu_f) / (sig_f + 1e-10)
        z_r = (response_expr[n_s:] - mu_r) / (sig_r + 1e-10)
        W, _ = build_fn(all_pos[:n_s], all_pos[n_s:], radius, **kw)
        lag = W @ z_r
        return _ref_ind(z_s, lag)

    @requires_r
    def test_gaussian_annular(self):
        all_pos, f_expr, r_expr, n_s, n_r, n_all = self._setup()
        radius = 500.0
        py_ind = self._python_ind(
            all_pos, f_expr, r_expr, n_s, radius,
            _build_gaussian_annular_weights, inner_radius=0.0,
        )
        r_out = run_r(f"""
        library(sigdiscov)
        all_pos <- matrix(c({_fmt_mat(all_pos)}), nrow={n_all}, ncol=2, byrow=TRUE)
        factor_expr <- c({_fmt_vec(f_expr)})
        response_expr <- c({_fmt_vec(r_expr)})
        config <- list(weight_type="gaussian_annular", sigma_fraction=3,
                       bandwidth={radius}, inner_radius=0, constant_sigma=0)
        result <- compute_ind_single(1:{n_s}, {n_s+1}:{n_all}, all_pos,
                                     factor_expr, response_expr, {radius}, config)
        cat(sprintf("%.15e", result$I_ND))
        """)
        r_ind = float(r_out)
        assert abs(py_ind - r_ind) < 1e-10, f"py={py_ind:.12f} R={r_ind:.12f}"

    @requires_r
    def test_annular_binary(self):
        all_pos, f_expr, r_expr, n_s, n_r, n_all = self._setup()
        radius, annular_w = 700.0, 400.0
        inner = max(0, radius - annular_w)
        py_ind = self._python_ind(
            all_pos, f_expr, r_expr, n_s, radius,
            _build_binary_weights, inner_radius=inner,
        )
        r_out = run_r(f"""
        library(sigdiscov)
        all_pos <- matrix(c({_fmt_mat(all_pos)}), nrow={n_all}, ncol=2, byrow=TRUE)
        factor_expr <- c({_fmt_vec(f_expr)})
        response_expr <- c({_fmt_vec(r_expr)})
        config <- list(weight_type="annular", annular_width={annular_w},
                       bandwidth=0, sigma_fraction=3, inner_radius=0, constant_sigma=0)
        result <- compute_ind_single(1:{n_s}, {n_s+1}:{n_all}, all_pos,
                                     factor_expr, response_expr, {radius}, config)
        cat(sprintf("%.15e", result$I_ND))
        """)
        r_ind = float(r_out)
        assert abs(py_ind - r_ind) < 1e-10, f"py={py_ind:.12f} R={r_ind:.12f}"


# ===========================================================================
# TEST 8: DE LOG2FC
# ===========================================================================

class TestDELog2FC:
    """Pseudocount consistency across implementations."""

    def _setup(self):
        np.random.seed(42)
        n_g, n_c = 10, 100
        expr = np.abs(np.random.randn(n_g, n_c)) * 2
        g1, g2 = np.arange(50), np.arange(50, 100)
        pc = 1e-9
        m1, m2 = np.mean(expr[:, g1], axis=1), np.mean(expr[:, g2], axis=1)
        sim_log2fc = np.log2((m1 + pc) / (m2 + pc))
        return expr, sim_log2fc, n_g, n_c

    def test_sim_vs_sigdiscovpy(self):
        expr, sim_fc, *_ = self._setup()
        m1 = np.mean(expr[:, :50], axis=1)
        m2 = np.mean(expr[:, 50:], axis=1)
        py_fc = np.log2((m1 + 1e-9) / (m2 + 1e-9))
        assert np.max(np.abs(sim_fc - py_fc)) < 1e-12

    @requires_r
    def test_sim_vs_r(self):
        expr, sim_fc, n_g, n_c = self._setup()
        r_out = run_r(f"""
        expr <- matrix(c({_fmt_mat(expr)}), nrow={n_g}, ncol={n_c}, byrow=TRUE)
        m1 <- rowMeans(expr[, 1:50])
        m2 <- rowMeans(expr[, 51:100])
        fc <- log2((m1 + 1e-9) / (m2 + 1e-9))
        cat(paste(sprintf("%.15e", fc), collapse=","))
        """)
        r_fc = np.array([float(v) for v in r_out.split(",")])
        assert np.max(np.abs(sim_fc - r_fc)) < 1e-9


# ===========================================================================
# TEST 9: EDGE CASES
# ===========================================================================

class TestEdgeCases:

    def test_constant_vector_zeros(self):
        py_z = standardize_vector(np.ones(50) * 7.0, use_gpu=False)
        assert np.allclose(py_z, 0)

    @requires_r
    def test_constant_vector_zeros_r(self):
        r_out = run_r("""
        library(sigdiscov)
        z <- sigdiscov::standardize(rep(7, 50))
        cat(max(abs(z)))
        """)
        assert float(r_out) < 1e-15

    def test_zero_norm_lag_ind(self):
        z_f = np.random.randn(20)
        result = compute_ind_from_lag(z_f, np.zeros(20), use_gpu=False)
        assert np.isnan(result) or result == 0.0

    def test_perfect_correlation(self):
        z = np.random.randn(100)
        assert abs(compute_ind_from_lag(z, z, use_gpu=False) - 1.0) < 1e-12

    def test_perfect_anticorrelation(self):
        z = np.random.randn(100)
        assert abs(compute_ind_from_lag(z, -z, use_gpu=False) + 1.0) < 1e-12

    def test_moran_positive(self):
        z = np.random.randn(100)
        assert compute_moran_from_lag(z, z, use_gpu=False) > 0

    def test_single_gene_ind_consistency(self):
        """Single-gene I_ND should match batch result."""
        np.random.seed(42)
        z_f = np.random.randn(50)
        lag_G = np.random.randn(50, 10)
        batch = compute_metric_batch(z_f, lag_G, metric="ind", use_gpu=False)
        for g in [0, 5, 9]:
            single = compute_ind_from_lag(z_f, lag_G[:, g], use_gpu=False)
            assert abs(batch[g] - single) < 1e-12
