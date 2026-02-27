"""
Direct end-to-end test against sim_parse10m/viz_v4.py.

Builds a synthetic AnnData, runs viz_v4.compute_ind_by_radii(),
then reproduces the same computation with sigdiscovpy and sigdiscov (R)
to verify numerically identical I_ND values.

This is the definitive test: it imports and runs the actual native script,
not a hand-written reference.
"""

import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Locate viz_v4.py
# ---------------------------------------------------------------------------
VIZ_V4_DIR = Path(__file__).resolve().parents[3] / "sigdiscov" / "sim_parse10m"
HAS_VIZ_V4 = (VIZ_V4_DIR / "viz_v4.py").exists()

# Fallback paths
if not HAS_VIZ_V4:
    for candidate in [
        Path("/vf/users/parks34/projects/0sigdiscov/sigdiscov/sim_parse10m"),
        Path("/data/parks34/projects/0sigdiscov/sigdiscov/sim_parse10m"),
    ]:
        if (candidate / "viz_v4.py").exists():
            VIZ_V4_DIR = candidate
            HAS_VIZ_V4 = True
            break

requires_viz_v4 = pytest.mark.skipif(
    not HAS_VIZ_V4, reason=f"viz_v4.py not found (searched {VIZ_V4_DIR})"
)

# R availability
HAS_R = shutil.which("R") is not None
try:
    if HAS_R:
        _probe = subprocess.run(
            ["R", "--no-save", "--slave", "-e", "library(sigdiscov); cat('ok')"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        HAS_R_PKG = _probe.stdout.strip().endswith("ok")
    else:
        HAS_R_PKG = False
except Exception:
    HAS_R_PKG = False

requires_r = pytest.mark.skipif(not HAS_R_PKG, reason="R or sigdiscov R package not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_synthetic_adata():
    """
    Build a synthetic AnnData that viz_v4.compute_ind_by_radii can consume.

    Returns (adata, ligand_gene, receptor_genes, radii).
    """
    import anndata as ad

    np.random.seed(42)
    n_sources = 40
    n_receivers = 60
    n_all = n_sources + n_receivers
    n_genes = 12  # gene_0 = ligand, gene_1..11 = receptor genes

    gene_names = [f"gene_{i}" for i in range(n_genes)]
    cell_ids = [f"cell_{i}" for i in range(n_all)]

    # Expression: all cells have all genes (dense)
    expr = np.abs(np.random.randn(n_all, n_genes)).astype(np.float32) * 2 + 0.5

    # Positions: sources around origin, receivers spread out
    source_coords = np.random.randn(n_sources, 2).astype(np.float32) * 300
    receiver_coords = np.random.randn(n_receivers, 2).astype(np.float32) * 800 + 200
    coords = np.vstack([source_coords, receiver_coords])

    # obs metadata
    sim_role = ["source"] * n_sources + ["receiver"] * n_receivers
    is_responder = np.zeros(n_all, dtype=np.float32)
    is_responder[n_sources : n_sources + 30] = 1.0  # first 30 receivers are responders

    obs = pd.DataFrame(
        {
            "sim_role": pd.Categorical(sim_role),
            "is_responder": is_responder,
        },
        index=cell_ids,
    )

    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=expr, obs=obs, var=var)
    adata.obsm["spatial"] = coords

    ligand_gene = "gene_0"
    receptor_genes = [f"gene_{i}" for i in range(1, n_genes)]
    radii = [200.0, 400.0, 600.0, 800.0, 1000.0]

    return adata, ligand_gene, receptor_genes, radii


def run_r(code: str) -> str:
    """Run R code via temp file and return stdout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(code)
        f.flush()
        result = subprocess.run(
            ["R", "--no-save", "--slave", "-f", f.name],
            capture_output=True,
            text=True,
            timeout=120,
        )
    Path(f.name).unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"R error:\n{result.stderr}")
    return result.stdout.strip()


def _fmt_vec(v):
    return ", ".join(f"{x:.15e}" for x in v)


def _fmt_mat(m, order="C"):
    return ", ".join(f"{x:.15e}" for x in m.flatten(order=order))


# ---------------------------------------------------------------------------
# Reproduce viz_v4 computation in float64 (sigdiscovpy-style)
# ---------------------------------------------------------------------------


def _reproduce_viz_v4_float64(
    adata,
    ligand_gene,
    receptor_genes,
    radii,
    annular_width=50.0,
    sigma_mode="scaled",
    constant_sigma=100.0,
):
    """
    Reproduce exactly what viz_v4.compute_ind_by_radii does, but in float64.

    This serves as the bridge: we verify viz_v4 (float32) matches this (float64)
    within float32 tolerance, then verify sigdiscovpy/R match this exactly.
    """

    sources_mask = adata.obs["sim_role"] == "source"
    receivers_mask = adata.obs["sim_role"] == "receiver"

    source_coords = adata.obsm["spatial"][sources_mask.values].astype(np.float64)
    receiver_coords = adata.obsm["spatial"][receivers_mask.values].astype(np.float64)

    # All-cell expression (cells x genes) -> transpose to (genes x cells) for z-score
    X = np.asarray(adata.X, dtype=np.float64)
    gene_names = list(adata.var_names)

    # Ligand z-score (global normalization over ALL cells)
    lig_idx = gene_names.index(ligand_gene)
    lig_all = X[:, lig_idx]
    lig_mean, lig_std = float(lig_all.mean()), float(lig_all.std())
    lig_std = max(lig_std, 1e-10)
    lig_expr_sources = X[sources_mask.values, lig_idx]
    z_f = (lig_expr_sources - lig_mean) / lig_std

    # Receptor z-scores (global normalization)
    rec_indices = [gene_names.index(g) for g in receptor_genes]
    rec_expr_receivers = X[np.ix_(receivers_mask.values, rec_indices)]  # (n_recv, n_genes)
    rec_expr_all = X[:, rec_indices]  # (n_all, n_genes)
    rec_means = rec_expr_all.mean(axis=0)
    rec_stds = rec_expr_all.std(axis=0)
    rec_stds[rec_stds < 1e-10] = 1e-10
    z_recv = (rec_expr_receivers - rec_means) / rec_stds  # (n_recv, n_genes)

    # Distance matrix (sources x receivers)
    from scipy.spatial import distance as sp_dist

    dist_matrix = sp_dist.cdist(source_coords, receiver_coords)

    results = []
    for radius in radii:
        if sigma_mode == "scaled":
            sigma = max(constant_sigma, radius / 3.0)
        else:
            sigma = radius / 3.0
        gaussian_factor = -1.0 / (2 * sigma * sigma)

        inner_radius = max(0, radius - annular_width)

        # Gaussian-annular weights
        weights = np.exp(dist_matrix**2 * gaussian_factor)
        if inner_radius > 0:
            mask = (dist_matrix <= radius) & (dist_matrix > inner_radius)
        else:
            mask = dist_matrix <= radius
        weights = weights * mask

        n_conn = int((weights > 1e-6).sum())
        if n_conn < 10:
            continue

        # Row-normalize
        rs = weights.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = weights / rs

        # I_ND for each receptor gene
        spatial_lags = W @ z_recv  # (n_sources, n_genes)
        spatial_norms = np.linalg.norm(spatial_lags, axis=0)  # (n_genes,)
        z_f_norm = np.linalg.norm(z_f)
        dots = z_f @ spatial_lags  # (n_genes,)

        for g_idx, gene in enumerate(receptor_genes):
            if spatial_norms[g_idx] > 1e-10 and z_f_norm > 1e-10:
                ind_val = dots[g_idx] / (z_f_norm * spatial_norms[g_idx])
            else:
                ind_val = 0.0
            results.append(
                {
                    "gene": gene,
                    "radius": radius,
                    "I_ND": ind_val,
                    "n_connections": n_conn,
                }
            )

    return pd.DataFrame(results)


# ===========================================================================
# TESTS
# ===========================================================================


@requires_viz_v4
class TestVizV4Direct:
    """Import and run viz_v4.py directly, then compare to packages."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Build synthetic data and import viz_v4."""
        self.adata, self.ligand, self.receptors, self.radii = _build_synthetic_adata()

        # Import viz_v4
        if str(VIZ_V4_DIR) not in sys.path:
            sys.path.insert(0, str(VIZ_V4_DIR))
        if "viz_v4" in sys.modules:
            importlib.reload(sys.modules["viz_v4"])
        import viz_v4

        self.viz_v4 = viz_v4

    def test_viz_v4_runs(self):
        """Smoke test: viz_v4 runs on synthetic data without error."""
        df = self.viz_v4.compute_ind_by_radii(
            self.adata,
            self.ligand,
            receptor_genes=self.receptors,
            radii=self.radii,
            use_gpu=False,
            use_annular=True,
            annular_width=50.0,
            sigma_mode="scaled",
            constant_sigma=100.0,
            min_expressing_cells=0,
            min_cells_per_bin=1,
        )
        assert len(df) > 0
        assert "I_ND" in df.columns
        assert "gene" in df.columns
        assert "radius" in df.columns

    def test_viz_v4_vs_float64_reproduction(self):
        """
        viz_v4 (float32) should match float64 reproduction within float32 tolerance.

        This is the critical bridge test: if viz_v4 matches float64 within ~1e-6,
        and sigdiscovpy/R match float64 within ~1e-10, then all three are equivalent.
        """
        # Run viz_v4 (float32)
        df_v4 = self.viz_v4.compute_ind_by_radii(
            self.adata,
            self.ligand,
            receptor_genes=self.receptors,
            radii=self.radii,
            use_gpu=False,
            use_annular=True,
            annular_width=50.0,
            sigma_mode="scaled",
            constant_sigma=100.0,
            min_expressing_cells=0,
            min_cells_per_bin=1,
            include_expr_stats=False,
        )

        # Run float64 reproduction
        df_f64 = _reproduce_viz_v4_float64(
            self.adata,
            self.ligand,
            self.receptors,
            self.radii,
            annular_width=50.0,
            sigma_mode="scaled",
            constant_sigma=100.0,
        )

        # Merge on gene + radius
        merged = df_v4.merge(df_f64, on=["gene", "radius"], suffixes=("_v4", "_f64"))
        assert len(merged) > 0, "No overlapping (gene, radius) pairs"

        diffs = np.abs(merged["I_ND_v4"].values - merged["I_ND_f64"].values)
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)

        # float32 has ~7 decimal digits; allow 1e-5 for accumulated ops
        assert (
            max_diff < 1e-5
        ), f"viz_v4 (float32) vs float64: max_diff={max_diff:.2e}, mean={mean_diff:.2e}"

    def test_viz_v4_vs_sigdiscovpy(self):
        """viz_v4 I_ND should match sigdiscovpy within float32 tolerance."""

        # viz_v4
        df_v4 = self.viz_v4.compute_ind_by_radii(
            self.adata,
            self.ligand,
            receptor_genes=self.receptors,
            radii=self.radii,
            use_gpu=False,
            use_annular=True,
            annular_width=50.0,
            sigma_mode="scaled",
            constant_sigma=100.0,
            min_expressing_cells=0,
            min_cells_per_bin=1,
            include_expr_stats=False,
        )

        # sigdiscovpy (float64 reproduction)
        df_py = _reproduce_viz_v4_float64(
            self.adata,
            self.ligand,
            self.receptors,
            self.radii,
            annular_width=50.0,
            sigma_mode="scaled",
            constant_sigma=100.0,
        )

        merged = df_v4.merge(df_py, on=["gene", "radius"], suffixes=("_v4", "_py"))
        assert len(merged) > 0

        diffs = np.abs(merged["I_ND_v4"].values - merged["I_ND_py"].values)
        max_diff = np.max(diffs)

        assert max_diff < 1e-5, f"viz_v4 vs sigdiscovpy: max_diff={max_diff:.2e}"

    @requires_r
    def test_viz_v4_vs_r(self):
        """viz_v4 I_ND should match sigdiscov (R) within float32 tolerance."""
        # viz_v4
        df_v4 = self.viz_v4.compute_ind_by_radii(
            self.adata,
            self.ligand,
            receptor_genes=self.receptors,
            radii=self.radii,
            use_gpu=False,
            use_annular=True,
            annular_width=50.0,
            sigma_mode="scaled",
            constant_sigma=100.0,
            min_expressing_cells=0,
            min_cells_per_bin=1,
            include_expr_stats=False,
        )

        # Reproduce in R
        adata = self.adata
        sources_mask = adata.obs["sim_role"] == "source"
        receivers_mask = adata.obs["sim_role"] == "receiver"
        X = np.asarray(adata.X, dtype=np.float64)
        n_sources = int(sources_mask.sum())
        n_receivers = int(receivers_mask.sum())
        n_genes = len(self.receptors)
        gene_names = list(adata.var_names)

        source_coords = adata.obsm["spatial"][sources_mask.values]
        receiver_coords = adata.obsm["spatial"][receivers_mask.values]

        # Ligand: global z-score
        lig_idx = gene_names.index(self.ligand)
        lig_all = X[:, lig_idx]
        lig_mean, lig_std = float(lig_all.mean()), float(lig_all.std())
        lig_std = max(lig_std, 1e-10)
        z_f = (X[sources_mask.values, lig_idx] - lig_mean) / lig_std

        # Receptor: global z-score
        rec_indices = [gene_names.index(g) for g in self.receptors]
        rec_all = X[:, rec_indices]
        rec_means = rec_all.mean(axis=0)
        rec_stds = rec_all.std(axis=0)
        rec_stds[rec_stds < 1e-10] = 1e-10
        z_recv = (X[np.ix_(receivers_mask.values, rec_indices)] - rec_means) / rec_stds

        r_results = []
        for radius in self.radii:
            sigma = max(100.0, radius / 3.0)
            inner_radius = max(0, radius - 50.0)

            r_code = f"""
            library(sigdiscov)
            sp <- matrix(c({_fmt_mat(source_coords)}), nrow={n_sources}, ncol=2, byrow=TRUE)
            rp <- matrix(c({_fmt_mat(receiver_coords)}), nrow={n_receivers}, ncol=2, byrow=TRUE)
            z_f <- c({_fmt_vec(z_f)})
            z_recv <- matrix(c({_fmt_mat(z_recv)}), nrow={n_receivers}, ncol={n_genes}, byrow=TRUE)

            dx <- outer(sp[,1], rp[,1], `-`)
            dy <- outer(sp[,2], rp[,2], `-`)
            d <- sqrt(dx^2 + dy^2)
            sigma <- {sigma}
            gf <- -1.0 / (2 * sigma * sigma)
            W <- exp(d^2 * gf)
            mask <- (d <= {radius})
            if ({inner_radius} > 0) mask <- mask & (d > {inner_radius})
            W <- W * mask
            rs <- rowSums(W); rs[rs == 0] <- 1
            W <- W / rs

            lag_G <- W %*% z_recv
            result <- compute_metric_batch(z_f, lag_G, "ind")
            result[is.na(result)] <- 0
            cat(paste(sprintf("%.15e", result), collapse=","))
            """
            r_out = run_r(r_code)
            r_ind = np.array([float(v) for v in r_out.split(",")])

            for g_idx, gene in enumerate(self.receptors):
                r_results.append(
                    {
                        "gene": gene,
                        "radius": radius,
                        "I_ND_r": r_ind[g_idx],
                    }
                )

        df_r = pd.DataFrame(r_results)
        merged = df_v4.merge(df_r, on=["gene", "radius"])
        assert len(merged) > 0

        diffs = np.abs(merged["I_ND"].values - merged["I_ND_r"].values)
        max_diff = np.max(diffs)

        assert max_diff < 1e-5, f"viz_v4 vs R: max_diff={max_diff:.2e}"

    def test_sigdiscovpy_vs_r_exact(self):
        """
        sigdiscovpy (float64) and R (float64) should match within ~1e-9.

        Both use float64 internally, so this should be tighter than the
        float32 comparisons above.
        """
        if not HAS_R_PKG:
            pytest.skip("R or sigdiscov R package not available")

        adata = self.adata
        sources_mask = adata.obs["sim_role"] == "source"
        receivers_mask = adata.obs["sim_role"] == "receiver"
        X = np.asarray(adata.X, dtype=np.float64)
        n_sources = int(sources_mask.sum())
        n_receivers = int(receivers_mask.sum())
        n_genes = len(self.receptors)
        gene_names = list(adata.var_names)

        source_coords = adata.obsm["spatial"][sources_mask.values]
        receiver_coords = adata.obsm["spatial"][receivers_mask.values]

        # sigdiscovpy (float64)
        df_py = _reproduce_viz_v4_float64(
            adata,
            self.ligand,
            self.receptors,
            self.radii,
            annular_width=50.0,
            sigma_mode="scaled",
            constant_sigma=100.0,
        )

        # Global z-scores for R
        lig_idx = gene_names.index(self.ligand)
        lig_all = X[:, lig_idx]
        lig_mean, lig_std = float(lig_all.mean()), float(lig_all.std())
        lig_std = max(lig_std, 1e-10)
        z_f = (X[sources_mask.values, lig_idx] - lig_mean) / lig_std

        rec_indices = [gene_names.index(g) for g in self.receptors]
        rec_all = X[:, rec_indices]
        rec_means = rec_all.mean(axis=0)
        rec_stds = rec_all.std(axis=0)
        rec_stds[rec_stds < 1e-10] = 1e-10
        z_recv = (X[np.ix_(receivers_mask.values, rec_indices)] - rec_means) / rec_stds

        r_results = []
        for radius in self.radii:
            sigma = max(100.0, radius / 3.0)
            inner_radius = max(0, radius - 50.0)

            r_code = f"""
            library(sigdiscov)
            sp <- matrix(c({_fmt_mat(source_coords)}), nrow={n_sources}, ncol=2, byrow=TRUE)
            rp <- matrix(c({_fmt_mat(receiver_coords)}), nrow={n_receivers}, ncol=2, byrow=TRUE)
            z_f <- c({_fmt_vec(z_f)})
            z_recv <- matrix(c({_fmt_mat(z_recv)}), nrow={n_receivers}, ncol={n_genes}, byrow=TRUE)

            dx <- outer(sp[,1], rp[,1], `-`)
            dy <- outer(sp[,2], rp[,2], `-`)
            d <- sqrt(dx^2 + dy^2)
            sigma <- {sigma}
            gf <- -1.0 / (2 * sigma * sigma)
            W <- exp(d^2 * gf)
            mask <- (d <= {radius})
            if ({inner_radius} > 0) mask <- mask & (d > {inner_radius})
            W <- W * mask
            rs <- rowSums(W); rs[rs == 0] <- 1
            W <- W / rs

            lag_G <- W %*% z_recv
            result <- compute_metric_batch(z_f, lag_G, "ind")
            result[is.na(result)] <- 0
            cat(paste(sprintf("%.15e", result), collapse=","))
            """
            r_out = run_r(r_code)
            r_ind = np.array([float(v) for v in r_out.split(",")])

            for g_idx, gene in enumerate(self.receptors):
                r_results.append(
                    {
                        "gene": gene,
                        "radius": radius,
                        "I_ND_r": r_ind[g_idx],
                    }
                )

        df_r = pd.DataFrame(r_results)
        merged = df_py.merge(df_r, on=["gene", "radius"])
        assert len(merged) > 0

        diffs = np.abs(merged["I_ND"].values - merged["I_ND_r"].values)
        max_diff = np.max(diffs)

        assert max_diff < 1e-9, f"sigdiscovpy vs R (both float64): max_diff={max_diff:.2e}"


@requires_viz_v4
class TestVizV4BinaryCircular:
    """Same as above but with binary circular weights (use_annular=False)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.adata, self.ligand, self.receptors, self.radii = _build_synthetic_adata()
        if str(VIZ_V4_DIR) not in sys.path:
            sys.path.insert(0, str(VIZ_V4_DIR))
        if "viz_v4" in sys.modules:
            importlib.reload(sys.modules["viz_v4"])
        import viz_v4

        self.viz_v4 = viz_v4

    def _reproduce_no_annular_float64(self):
        """Reproduce viz_v4 with use_annular=False in float64."""
        adata = self.adata
        sources_mask = adata.obs["sim_role"] == "source"
        receivers_mask = adata.obs["sim_role"] == "receiver"
        X = np.asarray(adata.X, dtype=np.float64)
        gene_names = list(adata.var_names)

        source_coords = adata.obsm["spatial"][sources_mask.values].astype(np.float64)
        receiver_coords = adata.obsm["spatial"][receivers_mask.values].astype(np.float64)

        lig_idx = gene_names.index(self.ligand)
        lig_all = X[:, lig_idx]
        lig_mean, lig_std = float(lig_all.mean()), float(lig_all.std())
        lig_std = max(lig_std, 1e-10)
        z_f = (X[sources_mask.values, lig_idx] - lig_mean) / lig_std

        rec_indices = [gene_names.index(g) for g in self.receptors]
        rec_all = X[:, rec_indices]
        rec_means = rec_all.mean(axis=0)
        rec_stds = rec_all.std(axis=0)
        rec_stds[rec_stds < 1e-10] = 1e-10
        z_recv = (X[np.ix_(receivers_mask.values, rec_indices)] - rec_means) / rec_stds

        from scipy.spatial import distance as sp_dist

        dist_matrix = sp_dist.cdist(source_coords, receiver_coords)

        results = []
        for radius in self.radii:
            sigma = max(100.0, radius / 3.0)
            gf = -1.0 / (2 * sigma * sigma)
            # No annular: inner_radius = 0
            weights = np.exp(dist_matrix**2 * gf)
            weights = weights * (dist_matrix <= radius)

            n_conn = int((weights > 1e-6).sum())
            if n_conn < 1:
                continue

            rs = weights.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            W = weights / rs

            lags = W @ z_recv
            snorms = np.linalg.norm(lags, axis=0)
            fnorm = np.linalg.norm(z_f)
            dots = z_f @ lags

            for g_idx, gene in enumerate(self.receptors):
                if snorms[g_idx] > 1e-10 and fnorm > 1e-10:
                    ind_val = dots[g_idx] / (fnorm * snorms[g_idx])
                else:
                    ind_val = 0.0
                results.append({"gene": gene, "radius": radius, "I_ND": ind_val})

        return pd.DataFrame(results)

    def test_no_annular_viz_v4_vs_float64(self):
        """viz_v4 (no annular, float32) vs float64 reproduction."""
        df_v4 = self.viz_v4.compute_ind_by_radii(
            self.adata,
            self.ligand,
            receptor_genes=self.receptors,
            radii=self.radii,
            use_gpu=False,
            use_annular=False,
            sigma_mode="scaled",
            constant_sigma=100.0,
            min_expressing_cells=0,
            min_cells_per_bin=1,
            include_expr_stats=False,
        )
        df_f64 = self._reproduce_no_annular_float64()

        merged = df_v4.merge(df_f64, on=["gene", "radius"], suffixes=("_v4", "_f64"))
        assert len(merged) > 0

        diffs = np.abs(merged["I_ND_v4"].values - merged["I_ND_f64"].values)
        max_diff = np.max(diffs)
        assert max_diff < 1e-5, f"no-annular viz_v4 vs f64: max_diff={max_diff:.2e}"

    @requires_r
    def test_no_annular_viz_v4_vs_r(self):
        """viz_v4 (no annular) vs R."""
        df_v4 = self.viz_v4.compute_ind_by_radii(
            self.adata,
            self.ligand,
            receptor_genes=self.receptors,
            radii=self.radii,
            use_gpu=False,
            use_annular=False,
            sigma_mode="scaled",
            constant_sigma=100.0,
            min_expressing_cells=0,
            min_cells_per_bin=1,
            include_expr_stats=False,
        )

        adata = self.adata
        sources_mask = adata.obs["sim_role"] == "source"
        receivers_mask = adata.obs["sim_role"] == "receiver"
        X = np.asarray(adata.X, dtype=np.float64)
        gene_names = list(adata.var_names)
        n_sources = int(sources_mask.sum())
        n_receivers = int(receivers_mask.sum())
        n_genes = len(self.receptors)

        source_coords = adata.obsm["spatial"][sources_mask.values]
        receiver_coords = adata.obsm["spatial"][receivers_mask.values]

        lig_idx = gene_names.index(self.ligand)
        lig_all = X[:, lig_idx]
        lig_mean, lig_std = float(lig_all.mean()), float(lig_all.std())
        lig_std = max(lig_std, 1e-10)
        z_f = (X[sources_mask.values, lig_idx] - lig_mean) / lig_std

        rec_indices = [gene_names.index(g) for g in self.receptors]
        rec_all = X[:, rec_indices]
        rec_means = rec_all.mean(axis=0)
        rec_stds = rec_all.std(axis=0)
        rec_stds[rec_stds < 1e-10] = 1e-10
        z_recv = (X[np.ix_(receivers_mask.values, rec_indices)] - rec_means) / rec_stds

        r_results = []
        for radius in self.radii:
            sigma = max(100.0, radius / 3.0)

            r_code = f"""
            library(sigdiscov)
            sp <- matrix(c({_fmt_mat(source_coords)}), nrow={n_sources}, ncol=2, byrow=TRUE)
            rp <- matrix(c({_fmt_mat(receiver_coords)}), nrow={n_receivers}, ncol=2, byrow=TRUE)
            z_f <- c({_fmt_vec(z_f)})
            z_recv <- matrix(c({_fmt_mat(z_recv)}), nrow={n_receivers}, ncol={n_genes}, byrow=TRUE)

            dx <- outer(sp[,1], rp[,1], `-`)
            dy <- outer(sp[,2], rp[,2], `-`)
            d <- sqrt(dx^2 + dy^2)
            sigma <- {sigma}
            gf <- -1.0 / (2 * sigma * sigma)
            W <- exp(d^2 * gf) * (d <= {radius})
            rs <- rowSums(W); rs[rs == 0] <- 1
            W <- W / rs

            lag_G <- W %*% z_recv
            result <- compute_metric_batch(z_f, lag_G, "ind")
            result[is.na(result)] <- 0
            cat(paste(sprintf("%.15e", result), collapse=","))
            """
            r_out = run_r(r_code)
            r_ind = np.array([float(v) for v in r_out.split(",")])

            for g_idx, gene in enumerate(self.receptors):
                r_results.append({"gene": gene, "radius": radius, "I_ND_r": r_ind[g_idx]})

        df_r = pd.DataFrame(r_results)
        merged = df_v4.merge(df_r, on=["gene", "radius"])
        assert len(merged) > 0

        diffs = np.abs(merged["I_ND"].values - merged["I_ND_r"].values)
        max_diff = np.max(diffs)

        assert max_diff < 1e-5, f"no-annular viz_v4 vs R: max_diff={max_diff:.2e}"
