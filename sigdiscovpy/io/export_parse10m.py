"""
Export Parse_10M h5ad data to CSV for R package consumption.

The R package (sigdiscov) does not load h5ad files directly.
This script extracts the required data into CSV files that R can read.

Usage:
    python -m sigdiscovpy.io.export_parse10m \\
        --h5ad path/to/parse10m.h5ad \\
        --cell-type "CD14 Mono" \\
        --cytokine "IFN-gamma" \\
        --output-dir ./exported_data
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def export_for_r(
    h5ad_path: str,
    cell_type: str,
    cytokine: str,
    output_dir: str,
    sender_type: Optional[str] = None,
    obs_celltype_key: str = "cell_type",
    obs_treatment_key: str = "treatment",
    coord_keys: tuple[str, str] = ("x", "y"),
    layer: Optional[str] = None,
) -> dict[str, str]:
    """
    Extract Parse_10M data to CSV files for R package consumption.

    Parameters
    ----------
    h5ad_path : str
        Path to h5ad file.
    cell_type : str
        Receiver cell type to filter (e.g., "CD14 Mono").
    cytokine : str
        Cytokine name for treatment filtering (e.g., "IFN-gamma").
    output_dir : str
        Output directory for CSV files.
    sender_type : str, optional
        Sender cell type. If None, uses all non-receiver cells with
        the cytokine treatment.
    obs_celltype_key : str
        Key in adata.obs for cell type labels.
    obs_treatment_key : str
        Key in adata.obs for treatment labels.
    coord_keys : tuple of str
        Keys in adata.obs or adata.obsm for x, y coordinates.
    layer : str, optional
        Layer to use for expression. If None, uses .X.

    Returns
    -------
    dict
        Paths to exported files: 'expression', 'metadata', 'coordinates'.
    """
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata is required: pip install anndata")

    try:
        import scipy.sparse
    except ImportError:
        scipy = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    logger.info(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Filter cells
    if obs_celltype_key in adata.obs.columns:
        cell_mask = adata.obs[obs_celltype_key] == cell_type
        logger.info(f"Cells of type '{cell_type}': {cell_mask.sum()}")
    else:
        logger.warning(
            f"Column '{obs_celltype_key}' not found. Using all cells."
        )
        cell_mask = pd.Series(True, index=adata.obs.index)

    # Filter by treatment if available
    if obs_treatment_key in adata.obs.columns:
        cyto_mask = adata.obs[obs_treatment_key].str.contains(cytokine, case=False, na=False)
        pbs_mask = adata.obs[obs_treatment_key].str.contains("PBS|control|untreated", case=False, na=False)
        logger.info(f"Cytokine-treated: {cyto_mask.sum()}, PBS/control: {pbs_mask.sum()}")
    else:
        cyto_mask = pd.Series(True, index=adata.obs.index)
        pbs_mask = pd.Series(False, index=adata.obs.index)

    # Select cells: receiver type with cytokine or PBS treatment
    selected = cell_mask & (cyto_mask | pbs_mask)
    adata_sub = adata[selected].copy()
    logger.info(f"Selected {adata_sub.shape[0]} cells after filtering")

    # Extract expression matrix (genes x cells)
    if layer and layer in adata_sub.layers:
        expr = adata_sub.layers[layer]
    else:
        expr = adata_sub.X

    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    expr = np.asarray(expr, dtype=np.float32)

    # Transpose to genes x cells
    expr_df = pd.DataFrame(
        expr.T,
        index=adata_sub.var_names,
        columns=adata_sub.obs_names,
    )

    # Extract metadata
    meta_df = pd.DataFrame({"cell_id": adata_sub.obs_names})
    if obs_celltype_key in adata_sub.obs.columns:
        meta_df["cell_type"] = adata_sub.obs[obs_celltype_key].values
    if obs_treatment_key in adata_sub.obs.columns:
        meta_df["treatment"] = adata_sub.obs[obs_treatment_key].values

    # Extract coordinates
    x_key, y_key = coord_keys
    if "spatial" in adata_sub.obsm:
        coords = adata_sub.obsm["spatial"]
        coord_df = pd.DataFrame(
            {"cell_id": adata_sub.obs_names, "x": coords[:, 0], "y": coords[:, 1]}
        )
    elif x_key in adata_sub.obs.columns and y_key in adata_sub.obs.columns:
        coord_df = pd.DataFrame(
            {
                "cell_id": adata_sub.obs_names,
                "x": adata_sub.obs[x_key].values,
                "y": adata_sub.obs[y_key].values,
            }
        )
    else:
        logger.warning("No spatial coordinates found. Exporting empty coordinates.")
        coord_df = pd.DataFrame(
            {"cell_id": adata_sub.obs_names, "x": 0.0, "y": 0.0}
        )

    # Save
    safe_ct = cell_type.replace(" ", "_").replace("/", "-")
    safe_cyto = cytokine.replace(" ", "_").replace("/", "-")
    prefix = f"{safe_ct}_{safe_cyto}"

    expr_path = output_dir / f"{prefix}_expression.csv"
    meta_path = output_dir / f"{prefix}_metadata.csv"
    coord_path = output_dir / f"{prefix}_coordinates.csv"

    logger.info(f"Saving expression matrix: {expr_df.shape}")
    expr_df.to_csv(expr_path)

    logger.info(f"Saving metadata: {meta_df.shape}")
    meta_df.to_csv(meta_path, index=False)

    logger.info(f"Saving coordinates: {coord_df.shape}")
    coord_df.to_csv(coord_path, index=False)

    logger.info(f"Export complete. Files in {output_dir}")

    return {
        "expression": str(expr_path),
        "metadata": str(meta_path),
        "coordinates": str(coord_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export Parse_10M h5ad data to CSV for R",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5ad", required=True, help="Path to h5ad file")
    parser.add_argument("--cell-type", required=True, help="Receiver cell type")
    parser.add_argument("--cytokine", required=True, help="Cytokine name")
    parser.add_argument("--output-dir", default="./exported_data", help="Output directory")
    parser.add_argument("--sender-type", default=None, help="Sender cell type")
    parser.add_argument("--celltype-key", default="cell_type", help="obs key for cell types")
    parser.add_argument("--treatment-key", default="treatment", help="obs key for treatment")
    parser.add_argument("--layer", default=None, help="Expression layer to use")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    export_for_r(
        h5ad_path=args.h5ad,
        cell_type=args.cell_type,
        cytokine=args.cytokine,
        output_dir=args.output_dir,
        sender_type=args.sender_type,
        obs_celltype_key=args.celltype_key,
        obs_treatment_key=args.treatment_key,
        layer=args.layer,
    )


if __name__ == "__main__":
    main()
