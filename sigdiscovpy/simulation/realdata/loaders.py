"""
Data loaders for real spatial transcriptomics datasets.

Supports loading from AnnData, CSV, and other common formats.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class SpatialData:
    """Container for spatial transcriptomics data."""

    expr: np.ndarray  # (n_genes, n_cells)
    coords: np.ndarray  # (n_cells, 2)
    gene_names: list[str]
    cell_types: Optional[np.ndarray] = None  # (n_cells,)
    cell_ids: Optional[list[str]] = None
    metadata: Optional[dict] = None

    @property
    def n_genes(self) -> int:
        return self.expr.shape[0]

    @property
    def n_cells(self) -> int:
        return self.expr.shape[1]


def load_anndata(
    path: Union[str, Path],
    layer: Optional[str] = None,
    coord_keys: tuple[str, str] = ("x", "y"),
    celltype_key: Optional[str] = "cell_type",
) -> SpatialData:
    """
    Load spatial data from AnnData (h5ad) file.

    Parameters
    ----------
    path : str or Path
        Path to .h5ad file.
    layer : str, optional
        Layer to use for expression. If None, uses .X.
    coord_keys : tuple
        Keys for x, y coordinates in .obs or .obsm['spatial'].
    celltype_key : str, optional
        Key for cell type annotations in .obs.

    Returns
    -------
    SpatialData
        Loaded spatial data.
    """
    import anndata

    adata = anndata.read_h5ad(path)

    # Get expression matrix
    if layer is not None:
        expr = adata.layers[layer]
    else:
        expr = adata.X

    # Convert to dense if sparse
    if hasattr(expr, "toarray"):
        expr = expr.toarray()

    # Transpose to (n_genes, n_cells)
    expr = expr.T.astype(np.float64)

    # Get coordinates
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"][:, :2].astype(np.float64)
    elif coord_keys[0] in adata.obs.columns:
        coords = adata.obs[[coord_keys[0], coord_keys[1]]].values.astype(np.float64)
    else:
        raise ValueError(
            f"Could not find coordinates. Tried 'spatial' in obsm and {coord_keys} in obs."
        )

    # Get cell types
    cell_types = None
    if celltype_key and celltype_key in adata.obs.columns:
        cell_types = adata.obs[celltype_key].values

    return SpatialData(
        expr=expr,
        coords=coords,
        gene_names=list(adata.var_names),
        cell_types=cell_types,
        cell_ids=list(adata.obs_names),
        metadata={"source": str(path), "n_obs": adata.n_obs, "n_vars": adata.n_vars},
    )


def load_csv(
    expr_path: Union[str, Path],
    coords_path: Union[str, Path],
    celltype_path: Optional[Union[str, Path]] = None,
    gene_column: Optional[str] = None,
    transpose: bool = False,
) -> SpatialData:
    """
    Load spatial data from CSV files.

    Parameters
    ----------
    expr_path : str or Path
        Path to expression matrix CSV.
    coords_path : str or Path
        Path to coordinates CSV with columns 'x', 'y' or first two columns.
    celltype_path : str or Path, optional
        Path to cell type annotations CSV.
    gene_column : str, optional
        Column name containing gene names. If None, uses index.
    transpose : bool
        If True, transpose expression matrix (cells x genes -> genes x cells).

    Returns
    -------
    SpatialData
        Loaded spatial data.
    """
    # Load expression
    expr_df = pd.read_csv(expr_path, index_col=0)

    if transpose:
        expr_df = expr_df.T

    expr = expr_df.values.astype(np.float64)
    gene_names = list(expr_df.index)
    cell_ids = list(expr_df.columns)

    # Load coordinates
    coords_df = pd.read_csv(coords_path, index_col=0)
    if "x" in coords_df.columns and "y" in coords_df.columns:
        coords = coords_df[["x", "y"]].values.astype(np.float64)
    else:
        coords = coords_df.iloc[:, :2].values.astype(np.float64)

    # Load cell types
    cell_types = None
    if celltype_path:
        ct_df = pd.read_csv(celltype_path, index_col=0)
        cell_types = ct_df.iloc[:, 0].values

    return SpatialData(
        expr=expr,
        coords=coords,
        gene_names=gene_names,
        cell_types=cell_types,
        cell_ids=cell_ids,
    )


def load_cosmx(
    data_dir: Union[str, Path],
    fov: Optional[int] = None,
) -> SpatialData:
    """
    Load CosMx data from Nanostring output directory.

    Parameters
    ----------
    data_dir : str or Path
        Path to CosMx output directory.
    fov : int, optional
        Specific field of view to load. If None, loads all.

    Returns
    -------
    SpatialData
        Loaded spatial data.
    """
    data_dir = Path(data_dir)

    # Find expression matrix
    expr_file = data_dir / "exprMat_file.csv"
    if not expr_file.exists():
        expr_file = list(data_dir.glob("*exprMat*.csv"))[0]

    # Find metadata
    meta_file = data_dir / "metadata_file.csv"
    if not meta_file.exists():
        meta_file = list(data_dir.glob("*metadata*.csv"))[0]

    # Load expression
    expr_df = pd.read_csv(expr_file, index_col=0)
    meta_df = pd.read_csv(meta_file, index_col=0)

    # Filter by FOV if specified
    if fov is not None:
        mask = meta_df["fov"] == fov
        expr_df = expr_df.loc[mask]
        meta_df = meta_df.loc[mask]

    # Get coordinates (CosMx uses CenterX_local_px, CenterY_local_px)
    x_col = [c for c in meta_df.columns if "CenterX" in c or "x_centroid" in c.lower()][0]
    y_col = [c for c in meta_df.columns if "CenterY" in c or "y_centroid" in c.lower()][0]
    coords = meta_df[[x_col, y_col]].values.astype(np.float64)

    # Get cell types if available
    cell_types = None
    ct_cols = [c for c in meta_df.columns if "cell_type" in c.lower() or "celltype" in c.lower()]
    if ct_cols:
        cell_types = meta_df[ct_cols[0]].values

    # Expression is cells x genes, transpose to genes x cells
    expr = expr_df.values.T.astype(np.float64)

    return SpatialData(
        expr=expr,
        coords=coords,
        gene_names=list(expr_df.columns),
        cell_types=cell_types,
        cell_ids=list(expr_df.index),
        metadata={"source": str(data_dir), "fov": fov},
    )


def subset_by_celltype(
    data: SpatialData,
    cell_types: list[str],
) -> SpatialData:
    """
    Subset data to specific cell types.

    Parameters
    ----------
    data : SpatialData
        Input data.
    cell_types : list of str
        Cell types to keep.

    Returns
    -------
    SpatialData
        Subsetted data.
    """
    if data.cell_types is None:
        raise ValueError("Data does not have cell type annotations")

    mask = np.isin(data.cell_types, cell_types)

    return SpatialData(
        expr=data.expr[:, mask],
        coords=data.coords[mask],
        gene_names=data.gene_names,
        cell_types=data.cell_types[mask],
        cell_ids=[data.cell_ids[i] for i in np.where(mask)[0]] if data.cell_ids else None,
        metadata=data.metadata,
    )


def filter_genes(
    data: SpatialData,
    min_cells: int = 10,
    min_counts: Optional[float] = None,
) -> SpatialData:
    """
    Filter genes by expression criteria.

    Parameters
    ----------
    data : SpatialData
        Input data.
    min_cells : int
        Minimum number of cells with non-zero expression.
    min_counts : float, optional
        Minimum total counts across cells.

    Returns
    -------
    SpatialData
        Filtered data.
    """
    # Count cells with expression
    n_cells_expressing = np.sum(data.expr > 0, axis=1)
    mask = n_cells_expressing >= min_cells

    if min_counts is not None:
        total_counts = np.sum(data.expr, axis=1)
        mask = mask & (total_counts >= min_counts)

    return SpatialData(
        expr=data.expr[mask],
        coords=data.coords,
        gene_names=[data.gene_names[i] for i in np.where(mask)[0]],
        cell_types=data.cell_types,
        cell_ids=data.cell_ids,
        metadata=data.metadata,
    )
