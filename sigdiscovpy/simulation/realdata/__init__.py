"""Real data loaders and processors for simulation validation."""

from sigdiscovpy.simulation.realdata.de_filter import (
    DEResult,
    filter_by_celltype_de,
    rank_genes_by_expression,
    wilcoxon_test,
)
from sigdiscovpy.simulation.realdata.loaders import (
    SpatialData,
    filter_genes,
    load_anndata,
    load_cosmx,
    load_csv,
    subset_by_celltype,
)
from sigdiscovpy.simulation.realdata.spatial_layout import (
    SpatialLayout,
    SpatialLayoutGenerator,
)

__all__ = [
    # Loaders
    "SpatialData",
    "load_anndata",
    "load_cosmx",
    "load_csv",
    "subset_by_celltype",
    "filter_genes",
    # DE Filter
    "DEResult",
    "wilcoxon_test",
    "filter_by_celltype_de",
    "rank_genes_by_expression",
    # Spatial Layout
    "SpatialLayout",
    "SpatialLayoutGenerator",
]
