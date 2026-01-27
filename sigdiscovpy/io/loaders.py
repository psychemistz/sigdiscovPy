"""
Data loading utilities for various spatial transcriptomics formats.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np


def load_anndata(
    path: str,
    layer: Optional[str] = None,
    obs_key: str = "cell_type",
) -> dict[str, Any]:
    """
    Load spatial data from AnnData (h5ad) file.

    Parameters
    ----------
    path : str
        Path to .h5ad file.
    layer : str, optional
        Layer to use for expression. If None, uses .X.
    obs_key : str, default="cell_type"
        Key in .obs for cell type annotations.

    Returns
    -------
    dict
        Dictionary with:
        - 'expr': Expression matrix (genes x cells)
        - 'coords': Spatial coordinates (cells x 2)
        - 'gene_names': Gene names
        - 'cell_types': Cell type annotations (if available)
        - 'adata': Original AnnData object

    Examples
    --------
    >>> data = load_anndata("data.h5ad")
    >>> data['expr'].shape
    (1000, 50000)  # genes x cells
    """
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata is required. Install with: pip install anndata")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    adata = ad.read_h5ad(path)

    # Get expression matrix
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}")
        expr = adata.layers[layer]
    else:
        expr = adata.X

    # Convert to dense if sparse
    from scipy import sparse

    if sparse.issparse(expr):
        expr = expr.toarray()

    # Transpose to genes x cells format
    expr = expr.T.astype(np.float64)

    # Get coordinates
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"].astype(np.float64)
    elif "X_spatial" in adata.obsm:
        coords = adata.obsm["X_spatial"].astype(np.float64)
    else:
        raise KeyError("No spatial coordinates found. Expected 'spatial' or 'X_spatial' in obsm.")

    # Get cell types
    cell_types = None
    if obs_key in adata.obs.columns:
        cell_types = adata.obs[obs_key].values

    return {
        "expr": expr,
        "coords": coords,
        "gene_names": adata.var_names.tolist(),
        "cell_names": adata.obs_names.tolist(),
        "cell_types": cell_types,
        "adata": adata,
    }


def load_cosmx(
    expr_path: str,
    coords_path: str,
    metadata_path: Optional[str] = None,
    cell_type_col: str = "cell_type",
) -> dict[str, Any]:
    """
    Load CosMx spatial transcriptomics data.

    Parameters
    ----------
    expr_path : str
        Path to expression matrix (CSV or TSV).
    coords_path : str
        Path to cell coordinates file.
    metadata_path : str, optional
        Path to cell metadata with cell types.
    cell_type_col : str, default="cell_type"
        Column name for cell types in metadata.

    Returns
    -------
    dict
        Dictionary with expr, coords, gene_names, cell_types.
    """
    import pandas as pd

    # Load expression
    expr_path = Path(expr_path)
    sep = "\t" if expr_path.suffix == ".tsv" else ","
    expr_df = pd.read_csv(expr_path, sep=sep, index_col=0)

    # Load coordinates
    coords_df = pd.read_csv(coords_path, sep=sep)
    if "x" in coords_df.columns and "y" in coords_df.columns:
        coords = coords_df[["x", "y"]].values
    elif "CenterX_global_px" in coords_df.columns:
        coords = coords_df[["CenterX_global_px", "CenterY_global_px"]].values
    else:
        raise KeyError("Cannot find coordinate columns in coords file.")

    # Expression as genes x cells
    expr = expr_df.values.astype(np.float64)
    gene_names = expr_df.index.tolist()

    # Load cell types if metadata provided
    cell_types = None
    if metadata_path is not None:
        meta_df = pd.read_csv(metadata_path, sep=sep)
        if cell_type_col in meta_df.columns:
            cell_types = meta_df[cell_type_col].values

    return {
        "expr": expr,
        "coords": coords.astype(np.float64),
        "gene_names": gene_names,
        "cell_types": cell_types,
    }


def load_coordinates(
    path: str,
    x_col: str = "x",
    y_col: str = "y",
) -> np.ndarray:
    """
    Load spatial coordinates from file.

    Parameters
    ----------
    path : str
        Path to coordinates file (CSV or TSV).
    x_col : str, default="x"
        Column name for x coordinates.
    y_col : str, default="y"
        Column name for y coordinates.

    Returns
    -------
    np.ndarray
        Coordinates array of shape (n, 2).
    """
    import pandas as pd

    path = Path(path)
    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)

    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(
            f"Columns '{x_col}' and/or '{y_col}' not found. " f"Available: {list(df.columns)}"
        )

    return df[[x_col, y_col]].values.astype(np.float64)


def parse_spot_names(spot_names: list) -> np.ndarray:
    """
    Parse spot coordinates from Visium "ROWxCOL" format names.

    Parameters
    ----------
    spot_names : list
        List of spot names in "ROWxCOL" format (e.g., ["34x56", "35x57"]).

    Returns
    -------
    np.ndarray
        Coordinates array of shape (n, 2) with [row, col] for each spot.

    Examples
    --------
    >>> spots = ["34x56", "35x57", "36x58"]
    >>> coords = parse_spot_names(spots)
    >>> coords
    array([[34, 56],
           [35, 57],
           [36, 58]])
    """
    coords = []
    for name in spot_names:
        name = str(name).strip()
        if "x" in name:
            parts = name.split("x")
            row, col = int(parts[0]), int(parts[1])
        else:
            raise ValueError(f"Invalid spot name format: '{name}'. Expected 'ROWxCOL'.")
        coords.append([row, col])

    return np.array(coords, dtype=np.float64)
