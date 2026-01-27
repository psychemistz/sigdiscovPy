"""Tests for realdata module."""

import numpy as np
import pytest

from sigdiscovpy.simulation.realdata import (
    SpatialData,
    SpatialLayout,
    SpatialLayoutGenerator,
    filter_genes,
    subset_by_celltype,
)
from sigdiscovpy.simulation.realdata.de_filter import (
    DEResult,
    rank_genes_by_expression,
    wilcoxon_test,
)


class TestSpatialData:
    """Tests for SpatialData container."""

    def test_basic_creation(self):
        """Test basic SpatialData creation."""
        expr = np.random.randn(10, 100)
        coords = np.random.randn(100, 2)
        gene_names = [f"gene_{i}" for i in range(10)]

        data = SpatialData(expr=expr, coords=coords, gene_names=gene_names)

        assert data.n_genes == 10
        assert data.n_cells == 100

    def test_with_cell_types(self):
        """Test SpatialData with cell types."""
        expr = np.random.randn(5, 50)
        coords = np.random.randn(50, 2)
        gene_names = [f"gene_{i}" for i in range(5)]
        cell_types = np.array(["A"] * 25 + ["B"] * 25)

        data = SpatialData(expr=expr, coords=coords, gene_names=gene_names, cell_types=cell_types)

        assert len(data.cell_types) == 50


class TestFilterFunctions:
    """Tests for data filtering functions."""

    def test_subset_by_celltype(self):
        """Test cell type subsetting."""
        expr = np.random.randn(5, 100)
        coords = np.random.randn(100, 2)
        gene_names = [f"gene_{i}" for i in range(5)]
        cell_types = np.array(["A"] * 40 + ["B"] * 30 + ["C"] * 30)

        data = SpatialData(expr=expr, coords=coords, gene_names=gene_names, cell_types=cell_types)

        subset = subset_by_celltype(data, ["A", "B"])

        assert subset.n_cells == 70
        assert set(subset.cell_types) == {"A", "B"}

    def test_filter_genes_by_cells(self):
        """Test gene filtering by minimum cells."""
        # Create expression with some genes expressed in few cells
        expr = np.zeros((10, 100))
        expr[:5, :50] = 1.0  # First 5 genes in 50 cells
        expr[5:, :5] = 1.0  # Last 5 genes in only 5 cells

        coords = np.random.randn(100, 2)
        gene_names = [f"gene_{i}" for i in range(10)]

        data = SpatialData(expr=expr, coords=coords, gene_names=gene_names)

        filtered = filter_genes(data, min_cells=10)

        assert filtered.n_genes == 5


class TestDEFilter:
    """Tests for differential expression filtering."""

    def test_wilcoxon_basic(self):
        """Test basic Wilcoxon test."""
        rng = np.random.default_rng(42)
        n_genes = 10
        n_cells = 100

        # Create expression with one DE gene
        expr = rng.lognormal(0, 0.5, (n_genes, n_cells))
        expr[0, :50] *= 5  # Gene 0 upregulated in group 1

        group1_mask = np.array([True] * 50 + [False] * 50)
        group2_mask = ~group1_mask
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        result = wilcoxon_test(expr, group1_mask, group2_mask, gene_names, fdr_threshold=0.1)

        assert isinstance(result, DEResult)
        assert len(result.pvalues) == n_genes
        assert result.pvalues[0] < 0.05  # DE gene should be significant

    def test_rank_genes(self):
        """Test gene ranking by expression."""
        rng = np.random.default_rng(42)
        n_genes = 20
        n_cells = 50

        expr = rng.lognormal(0, 1, (n_genes, n_cells))
        # Make gene 0 highly expressed
        expr[0, :] *= 10

        cell_mask = np.ones(n_cells, dtype=bool)
        gene_names = [f"gene_{i}" for i in range(n_genes)]

        top_genes = rank_genes_by_expression(expr, cell_mask, gene_names, top_n=5)

        assert len(top_genes) == 5
        assert "gene_0" in top_genes  # Should be in top genes


class TestSpatialLayoutGenerator:
    """Tests for SpatialLayoutGenerator."""

    def test_extract_layout(self):
        """Test layout extraction."""
        expr = np.random.randn(10, 100)
        coords = np.random.randn(100, 2) * 100
        gene_names = [f"gene_{i}" for i in range(10)]
        cell_types = np.array(["sender"] * 20 + ["receiver"] * 30 + ["other"] * 50)

        data = SpatialData(expr=expr, coords=coords, gene_names=gene_names, cell_types=cell_types)

        generator = SpatialLayoutGenerator(data)
        layout = generator.extract_layout("sender", "receiver")

        assert isinstance(layout, SpatialLayout)
        assert layout.n_senders == 20
        assert layout.n_receivers == 30

    def test_celltype_fractions(self):
        """Test cell type fraction computation."""
        expr = np.random.randn(5, 100)
        coords = np.random.randn(100, 2)
        gene_names = [f"gene_{i}" for i in range(5)]
        cell_types = np.array(["A"] * 40 + ["B"] * 60)

        data = SpatialData(expr=expr, coords=coords, gene_names=gene_names, cell_types=cell_types)

        generator = SpatialLayoutGenerator(data)
        fractions = generator.compute_celltype_fractions()

        assert fractions["A"] == pytest.approx(0.4)
        assert fractions["B"] == pytest.approx(0.6)

    def test_generate_similar_layout(self):
        """Test generating similar layout."""
        expr = np.random.randn(5, 100)
        coords = np.random.randn(100, 2) * 100
        gene_names = [f"gene_{i}" for i in range(5)]
        cell_types = np.array(["A"] * 50 + ["B"] * 50)

        data = SpatialData(expr=expr, coords=coords, gene_names=gene_names, cell_types=cell_types)

        generator = SpatialLayoutGenerator(data)
        new_layout = generator.generate_similar_layout(
            n_cells=200, sender_fraction=0.2, receiver_fraction=0.3, seed=42
        )

        assert len(new_layout.coords) == 200
        assert new_layout.n_senders == 40
        assert new_layout.n_receivers == 60
