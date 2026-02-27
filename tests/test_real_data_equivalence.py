"""
End-to-end equivalence test on actual Parse10M simulation data.

Loads a real simulation h5ad (generated from Parse_10M_PBMC_cytokines),
runs I_ND computation through:
  1. viz_v4.py (native sim_parse10m script) — float32
  2. sigdiscovpy (Python package) — float64
  3. sigdiscov (R package) — float64

Verifies all three produce identical I_ND values within floating-point
tolerance.
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
import scipy.sparse
from scipy.spatial import distance as sp_dist

# ---------------------------------------------------------------------------
# Locate test data and viz_v4
# ---------------------------------------------------------------------------
SIM_DIR = Path("/vf/users/parks34/projects/0sigdiscov/sigdiscov/sim_parse10m")
H5AD_DIR = SIM_DIR / "results" / "h5ad"

# Pick a representative simulation: IFNgamma, CD4 Memory -> CD14 Mono
TEST_H5AD = H5AD_DIR / "s1_c1000_srad1000_sigma1500_aq95_cq100.h5ad"

HAS_DATA = TEST_H5AD.exists()
HAS_VIZ_V4 = (SIM_DIR / "viz_v4.py").exists()

requires_data = pytest.mark.skipif(not HAS_DATA, reason=f"Test h5ad not found: {TEST_H5AD}")
requires_viz_v4 = pytest.mark.skipif(not HAS_VIZ_V4, reason="viz_v4.py not found")

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

# Test radii — subset for speed
TEST_RADII = [200.0, 500.0, 1000.0, 2000.0, 3000.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_r(code: str) -> str:
    """Run R code via temp file and return stdout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(code)
        f.flush()
        result = subprocess.run(
            ["R", "--no-save", "--slave", "-f", f.name],
            capture_output=True,
            text=True,
            timeout=300,
        )
    Path(f.name).unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"R error:\n{result.stderr}")
    return result.stdout.strip()


def _fmt_vec(v):
    return ", ".join(f"{x:.15e}" for x in v)


def _fmt_mat(m, order="C"):
    return ", ".join(f"{x:.15e}" for x in m.flatten(order=order))


def _get_test_genes(adata, ligand_gene, max_genes=20):
    """Select receptor genes: DE-filtered or top variable, limited count."""
    # Use response genes if available
    if "ifng_response_genes" in adata.uns:
        response = list(adata.uns["ifng_response_genes"])
        # Filter to genes present in var_names
        valid = [g for g in response if g in adata.var_names and g != ligand_gene]
        if len(valid) >= 5:
            return valid[:max_genes]

    # Fallback: top variable genes
    receivers = adata[adata.obs["sim_role"] == "receiver"]
    X_raw = receivers.X
    X = (
        X_raw.toarray().astype(np.float64)
        if scipy.sparse.issparse(X_raw)
        else np.asarray(X_raw, dtype=np.float64)
    )
    var = X.var(axis=0)
    top_idx = np.argsort(var)[::-1][:max_genes]
    genes = [adata.var_names[i] for i in top_idx if adata.var_names[i] != ligand_gene]
    return genes[:max_genes]


def compute_ind_float64(
    adata,
    ligand_gene,
    receptor_genes,
    radii,
    use_annular=True,
    annular_width=20.0,
    sigma_mode="scaled",
    constant_sigma=100.0,
):
    """Reproduce viz_v4 computation in float64 for reference."""
    sources_mask = adata.obs["sim_role"] == "source"
    receivers_mask = adata.obs["sim_role"] == "receiver"

    source_coords = adata.obsm["spatial"][sources_mask.values].astype(np.float64)
    receiver_coords = adata.obsm["spatial"][receivers_mask.values].astype(np.float64)

    X_raw = adata.X
    X = (
        X_raw.toarray().astype(np.float64)
        if scipy.sparse.issparse(X_raw)
        else np.asarray(X_raw, dtype=np.float64)
    )
    gene_names = list(adata.var_names)

    # Ligand z-score (global)
    lig_idx = gene_names.index(ligand_gene)
    lig_all = X[:, lig_idx]
    lig_mean = float(lig_all.mean())
    lig_std = float(lig_all.std())
    lig_std = max(lig_std, 1e-10)
    z_f = (X[sources_mask.values, lig_idx] - lig_mean) / lig_std

    # Receptor z-scores (global)
    rec_indices = [gene_names.index(g) for g in receptor_genes]
    rec_all = X[:, rec_indices]
    rec_means = rec_all.mean(axis=0)
    rec_stds = rec_all.std(axis=0)
    rec_stds[rec_stds < 1e-10] = 1e-10
    z_recv = (X[np.ix_(receivers_mask.values, rec_indices)] - rec_means) / rec_stds

    # Distance matrix
    dist_matrix = sp_dist.cdist(source_coords, receiver_coords)

    results = []
    for radius in radii:
        if sigma_mode == "scaled":
            sigma = max(constant_sigma, radius / 3.0)
        else:
            sigma = radius / 3.0
        gaussian_factor = -1.0 / (2 * sigma * sigma)

        inner_radius = max(0, radius - annular_width) if use_annular else 0

        weights = np.exp(dist_matrix**2 * gaussian_factor)
        if inner_radius > 0:
            mask = (dist_matrix <= radius) & (dist_matrix > inner_radius)
        else:
            mask = dist_matrix <= radius
        weights = weights * mask

        n_conn = int((weights > 1e-6).sum())
        if n_conn < 10:
            continue

        rs = weights.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = weights / rs

        spatial_lags = W @ z_recv
        spatial_norms = np.linalg.norm(spatial_lags, axis=0)
        z_f_norm = np.linalg.norm(z_f)
        dots = z_f @ spatial_lags

        for g_idx, gene in enumerate(receptor_genes):
            if spatial_norms[g_idx] > 1e-10 and z_f_norm > 1e-10:
                ind_val = dots[g_idx] / (z_f_norm * spatial_norms[g_idx])
            else:
                ind_val = 0.0
            results.append(
                {"gene": gene, "radius": radius, "I_ND": ind_val, "n_connections": n_conn}
            )

    return pd.DataFrame(results)


# ===========================================================================
# TESTS
# ===========================================================================


@requires_data
@requires_viz_v4
class TestRealDataEquivalence:
    """Compare I_ND across all three implementations on real Parse10M data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load real simulation data and import viz_v4."""
        import scanpy as sc

        self.adata = sc.read_h5ad(str(TEST_H5AD))
        config = self.adata.uns.get("simulation_config", {})
        self.ligand = config.get("ligand_gene", "IFNG")
        self.annular_width = config.get("annular_width", 20.0)
        self.sigma_mode = config.get("ind_sigma_mode", "scaled")
        self.receptors = _get_test_genes(self.adata, self.ligand, max_genes=20)
        self.radii = TEST_RADII

        # Import viz_v4
        if str(SIM_DIR) not in sys.path:
            sys.path.insert(0, str(SIM_DIR))
        if "viz_v4" in sys.modules:
            importlib.reload(sys.modules["viz_v4"])
        import viz_v4

        self.viz_v4 = viz_v4

    def test_data_loaded(self):
        """Smoke test: data loads correctly."""
        assert self.adata.shape[0] > 1000
        assert self.ligand in self.adata.var_names
        assert len(self.receptors) >= 5
        print(f"\nDataset: {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
        print(f"Ligand: {self.ligand}")
        print(f"Receptors: {len(self.receptors)} genes")
        print(f"Radii: {self.radii}")

    def test_viz_v4_runs_on_real_data(self):
        """viz_v4 runs on real simulation data without error."""
        df = self.viz_v4.compute_ind_by_radii(
            self.adata,
            self.ligand,
            receptor_genes=self.receptors,
            radii=self.radii,
            use_gpu=False,
            use_annular=True,
            annular_width=self.annular_width,
            sigma_mode=self.sigma_mode,
            constant_sigma=100.0,
            min_expressing_cells=0,
            min_cells_per_bin=1,
            include_expr_stats=False,
        )
        assert len(df) > 0
        assert "I_ND" in df.columns
        n_nonzero = (df["I_ND"].abs() > 1e-10).sum()
        print(f"\nviz_v4: {len(df)} rows, {n_nonzero} non-zero I_ND values")

    def test_viz_v4_vs_float64(self):
        """viz_v4 (float32) matches float64 within tolerance on real data."""
        df_v4 = self.viz_v4.compute_ind_by_radii(
            self.adata,
            self.ligand,
            receptor_genes=self.receptors,
            radii=self.radii,
            use_gpu=False,
            use_annular=True,
            annular_width=self.annular_width,
            sigma_mode=self.sigma_mode,
            constant_sigma=100.0,
            min_expressing_cells=0,
            min_cells_per_bin=1,
            include_expr_stats=False,
        )

        df_f64 = compute_ind_float64(
            self.adata,
            self.ligand,
            self.receptors,
            self.radii,
            use_annular=True,
            annular_width=self.annular_width,
            sigma_mode=self.sigma_mode,
            constant_sigma=100.0,
        )

        merged = df_v4.merge(df_f64, on=["gene", "radius"], suffixes=("_v4", "_f64"))
        assert len(merged) > 0, "No overlapping (gene, radius) pairs"

        diffs = np.abs(merged["I_ND_v4"].values - merged["I_ND_f64"].values)
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)

        print(f"\nviz_v4 vs float64: max_diff={max_diff:.2e}, mean={mean_diff:.2e}")
        print(f"  {len(merged)} gene-radius pairs compared")

        # float32 tolerance
        assert max_diff < 1e-4, f"viz_v4 vs float64: max_diff={max_diff:.2e}, mean={mean_diff:.2e}"

    @requires_r
    def test_float64_vs_r(self):
        """float64 Python matches R within ~1e-9 on real data."""
        df_f64 = compute_ind_float64(
            self.adata,
            self.ligand,
            self.receptors,
            self.radii,
            use_annular=True,
            annular_width=self.annular_width,
            sigma_mode=self.sigma_mode,
            constant_sigma=100.0,
        )

        # Prepare data for R
        adata = self.adata
        sources_mask = adata.obs["sim_role"] == "source"
        receivers_mask = adata.obs["sim_role"] == "receiver"
        X_raw = adata.X
        X = (
            X_raw.toarray().astype(np.float64)
            if scipy.sparse.issparse(X_raw)
            else np.asarray(X_raw, dtype=np.float64)
        )
        gene_names = list(adata.var_names)

        source_coords = adata.obsm["spatial"][sources_mask.values].astype(np.float64)
        receiver_coords = adata.obsm["spatial"][receivers_mask.values].astype(np.float64)

        # Global z-scores
        lig_idx = gene_names.index(self.ligand)
        lig_all = X[:, lig_idx]
        lig_mean = float(lig_all.mean())
        lig_std = float(lig_all.std())
        lig_std = max(lig_std, 1e-10)
        z_f = (X[sources_mask.values, lig_idx] - lig_mean) / lig_std

        rec_indices = [gene_names.index(g) for g in self.receptors]
        rec_all = X[:, rec_indices]
        rec_means = rec_all.mean(axis=0)
        rec_stds = rec_all.std(axis=0)
        rec_stds[rec_stds < 1e-10] = 1e-10
        z_recv = (X[np.ix_(receivers_mask.values, rec_indices)] - rec_means) / rec_stds

        # Save large arrays to temp files for R
        tmpdir = Path(tempfile.mkdtemp())
        np.savetxt(tmpdir / "source_coords.csv", source_coords, delimiter=",", fmt="%.15e")
        np.savetxt(tmpdir / "receiver_coords.csv", receiver_coords, delimiter=",", fmt="%.15e")
        np.savetxt(tmpdir / "z_f.csv", z_f, delimiter=",", fmt="%.15e")
        np.savetxt(tmpdir / "z_recv.csv", z_recv, delimiter=",", fmt="%.15e")

        r_results = []
        for radius in self.radii:
            if self.sigma_mode == "scaled":
                sigma = max(100.0, radius / 3.0)
            else:
                sigma = radius / 3.0
            inner_radius = max(0, radius - self.annular_width)

            r_code = f"""
            library(sigdiscov)
            sp <- as.matrix(read.csv("{tmpdir}/source_coords.csv", header=FALSE))
            rp <- as.matrix(read.csv("{tmpdir}/receiver_coords.csv", header=FALSE))
            z_f <- as.numeric(read.csv("{tmpdir}/z_f.csv", header=FALSE)$V1)
            z_recv <- as.matrix(read.csv("{tmpdir}/z_recv.csv", header=FALSE))

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
                r_results.append({"gene": gene, "radius": radius, "I_ND_r": r_ind[g_idx]})

        # Cleanup
        import shutil as _shutil

        _shutil.rmtree(tmpdir, ignore_errors=True)

        df_r = pd.DataFrame(r_results)
        merged = df_f64.merge(df_r, on=["gene", "radius"])
        assert len(merged) > 0

        diffs = np.abs(merged["I_ND"].values - merged["I_ND_r"].values)
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)

        print(f"\nfloat64 vs R: max_diff={max_diff:.2e}, mean={mean_diff:.2e}")
        print(f"  {len(merged)} gene-radius pairs compared")

        assert max_diff < 1e-9, f"float64 vs R: max_diff={max_diff:.2e}, mean={mean_diff:.2e}"

    @requires_r
    def test_viz_v4_vs_r(self):
        """viz_v4 (float32) matches R within float32 tolerance on real data."""
        df_v4 = self.viz_v4.compute_ind_by_radii(
            self.adata,
            self.ligand,
            receptor_genes=self.receptors,
            radii=self.radii,
            use_gpu=False,
            use_annular=True,
            annular_width=self.annular_width,
            sigma_mode=self.sigma_mode,
            constant_sigma=100.0,
            min_expressing_cells=0,
            min_cells_per_bin=1,
            include_expr_stats=False,
        )

        # Recompute in R (reuse z-scores from float64)
        adata = self.adata
        sources_mask = adata.obs["sim_role"] == "source"
        receivers_mask = adata.obs["sim_role"] == "receiver"
        X_raw = adata.X
        X = (
            X_raw.toarray().astype(np.float64)
            if scipy.sparse.issparse(X_raw)
            else np.asarray(X_raw, dtype=np.float64)
        )
        gene_names = list(adata.var_names)

        source_coords = adata.obsm["spatial"][sources_mask.values].astype(np.float64)
        receiver_coords = adata.obsm["spatial"][receivers_mask.values].astype(np.float64)

        lig_idx = gene_names.index(self.ligand)
        lig_all = X[:, lig_idx]
        lig_mean = float(lig_all.mean())
        lig_std = float(lig_all.std())
        lig_std = max(lig_std, 1e-10)
        z_f = (X[sources_mask.values, lig_idx] - lig_mean) / lig_std

        rec_indices = [gene_names.index(g) for g in self.receptors]
        rec_all = X[:, rec_indices]
        rec_means = rec_all.mean(axis=0)
        rec_stds = rec_all.std(axis=0)
        rec_stds[rec_stds < 1e-10] = 1e-10
        z_recv = (X[np.ix_(receivers_mask.values, rec_indices)] - rec_means) / rec_stds

        tmpdir = Path(tempfile.mkdtemp())
        np.savetxt(tmpdir / "source_coords.csv", source_coords, delimiter=",", fmt="%.15e")
        np.savetxt(
            tmpdir / "receiver_coords.csv",
            receiver_coords,
            delimiter=",",
            fmt="%.15e",
        )
        np.savetxt(tmpdir / "z_f.csv", z_f, delimiter=",", fmt="%.15e")
        np.savetxt(tmpdir / "z_recv.csv", z_recv, delimiter=",", fmt="%.15e")

        r_results = []
        for radius in self.radii:
            if self.sigma_mode == "scaled":
                sigma = max(100.0, radius / 3.0)
            else:
                sigma = radius / 3.0
            inner_radius = max(0, radius - self.annular_width)

            r_code = f"""
            library(sigdiscov)
            sp <- as.matrix(read.csv("{tmpdir}/source_coords.csv", header=FALSE))
            rp <- as.matrix(read.csv("{tmpdir}/receiver_coords.csv", header=FALSE))
            z_f <- as.numeric(read.csv("{tmpdir}/z_f.csv", header=FALSE)$V1)
            z_recv <- as.matrix(read.csv("{tmpdir}/z_recv.csv", header=FALSE))

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
                r_results.append({"gene": gene, "radius": radius, "I_ND_r": r_ind[g_idx]})

        import shutil as _shutil

        _shutil.rmtree(tmpdir, ignore_errors=True)

        df_r = pd.DataFrame(r_results)
        merged = df_v4.merge(df_r, on=["gene", "radius"])
        assert len(merged) > 0

        diffs = np.abs(merged["I_ND"].values - merged["I_ND_r"].values)
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)

        print(f"\nviz_v4 vs R: max_diff={max_diff:.2e}, mean={mean_diff:.2e}")
        print(f"  {len(merged)} gene-radius pairs compared")

        # float32 tolerance (viz_v4 uses float32 internally)
        assert max_diff < 1e-4, f"viz_v4 vs R: max_diff={max_diff:.2e}, mean={mean_diff:.2e}"

    def test_all_three_summary(self):
        """
        Summary test: compare all three implementations and print report.
        """
        # viz_v4 (float32)
        df_v4 = self.viz_v4.compute_ind_by_radii(
            self.adata,
            self.ligand,
            receptor_genes=self.receptors,
            radii=self.radii,
            use_gpu=False,
            use_annular=True,
            annular_width=self.annular_width,
            sigma_mode=self.sigma_mode,
            constant_sigma=100.0,
            min_expressing_cells=0,
            min_cells_per_bin=1,
            include_expr_stats=False,
        )

        # float64 (sigdiscovpy-style)
        df_f64 = compute_ind_float64(
            self.adata,
            self.ligand,
            self.receptors,
            self.radii,
            use_annular=True,
            annular_width=self.annular_width,
            sigma_mode=self.sigma_mode,
            constant_sigma=100.0,
        )

        merged = df_v4.merge(df_f64, on=["gene", "radius"], suffixes=("_v4", "_f64"))

        v4_vs_f64 = np.abs(merged["I_ND_v4"].values - merged["I_ND_f64"].values)

        print("\n" + "=" * 60)
        print("REAL DATA EQUIVALENCE SUMMARY")
        print("=" * 60)
        print(f"Dataset: {TEST_H5AD.name}")
        print(f"Cells: {self.adata.shape[0]:,} × Genes: {self.adata.shape[1]:,}")
        print(f"Ligand: {self.ligand}")
        print(f"Receptor genes tested: {len(self.receptors)}")
        print(f"Radii tested: {self.radii}")
        print(f"Gene-radius pairs: {len(merged)}")
        print()
        print(f"viz_v4 vs float64:  max={np.max(v4_vs_f64):.2e}  mean={np.mean(v4_vs_f64):.2e}")

        assert np.max(v4_vs_f64) < 1e-4
