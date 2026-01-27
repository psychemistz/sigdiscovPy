"""
HDF5 I/O utilities for storing and loading analysis results.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path


def save_results_hdf5(
    path: str,
    matrices: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, Any]] = None,
    gene_names: Optional[List[str]] = None,
    cell_types: Optional[List[str]] = None,
    radii: Optional[List[float]] = None,
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    """
    Save analysis results to HDF5 file.

    Parameters
    ----------
    path : str
        Output file path.
    matrices : dict
        Dictionary of arrays to save. Keys become dataset names.
    metadata : dict, optional
        Additional metadata as attributes.
    gene_names : list, optional
        Gene names to store.
    cell_types : list, optional
        Cell type names to store.
    radii : list, optional
        Radii values to store.
    compression : str, default="gzip"
        Compression algorithm.
    compression_level : int, default=4
        Compression level (1-9).

    Examples
    --------
    >>> matrices = {"morans_i": np.random.randn(100, 100)}
    >>> save_results_hdf5("results.h5", matrices, gene_names=gene_list)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required. Install with: pip install h5py")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Save matrices
        for name, arr in matrices.items():
            f.create_dataset(
                name,
                data=arr,
                compression=compression,
                compression_opts=compression_level,
            )

        # Save metadata as attributes
        if metadata is not None:
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    f.attrs[key] = value
                elif isinstance(value, (list, tuple)):
                    f.attrs[key] = np.array(value)

        # Save gene names
        if gene_names is not None:
            f.create_dataset(
                "gene_names",
                data=np.array(gene_names, dtype="S"),
            )

        # Save cell types
        if cell_types is not None:
            f.create_dataset(
                "cell_types",
                data=np.array(cell_types, dtype="S"),
            )

        # Save radii
        if radii is not None:
            f.create_dataset("radii", data=np.array(radii))


def load_results_hdf5(path: str) -> Dict[str, Any]:
    """
    Load analysis results from HDF5 file.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    dict
        Dictionary with:
        - Each dataset as a key-value pair
        - 'metadata': dict of file attributes
        - 'gene_names': list of gene names (if present)
        - 'cell_types': list of cell types (if present)
        - 'radii': array of radii (if present)

    Examples
    --------
    >>> results = load_results_hdf5("results.h5")
    >>> results['morans_i'].shape
    (100, 100)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required. Install with: pip install h5py")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    result = {}
    metadata = {}

    with h5py.File(path, "r") as f:
        # Load all datasets
        for key in f.keys():
            if key == "gene_names":
                result["gene_names"] = [x.decode() for x in f[key][:]]
            elif key == "cell_types":
                result["cell_types"] = [x.decode() for x in f[key][:]]
            elif key == "radii":
                result["radii"] = f[key][:]
            else:
                result[key] = f[key][:]

        # Load attributes as metadata
        for key, value in f.attrs.items():
            metadata[key] = value

    result["metadata"] = metadata
    return result


class HDF5ResultWriter:
    """
    Context manager for streaming results to HDF5.

    Use this for large-scale analysis where results are computed incrementally.

    Examples
    --------
    >>> with HDF5ResultWriter("results.h5", n_genes=1000, n_radii=10) as writer:
    ...     for r_idx, radius in enumerate(radii):
    ...         matrix = compute_matrix(radius)
    ...         writer.write_radius(r_idx, matrix)
    """

    def __init__(
        self,
        path: str,
        n_genes: int,
        n_radii: int,
        gene_names: Optional[List[str]] = None,
        radii: Optional[List[float]] = None,
        dtype: np.dtype = np.float32,
        compression: str = "gzip",
    ):
        self.path = Path(path)
        self.n_genes = n_genes
        self.n_radii = n_radii
        self.gene_names = gene_names
        self.radii = radii
        self.dtype = dtype
        self.compression = compression
        self.file = None

    def __enter__(self):
        import h5py

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "w")

        # Create main dataset
        self.file.create_dataset(
            "morans_i_matrices",
            shape=(self.n_radii, self.n_genes, self.n_genes),
            dtype=self.dtype,
            chunks=(1, min(100, self.n_genes), min(100, self.n_genes)),
            compression=self.compression,
        )

        # Save metadata
        if self.gene_names is not None:
            self.file.create_dataset(
                "gene_names",
                data=np.array(self.gene_names, dtype="S"),
            )

        if self.radii is not None:
            self.file.create_dataset("radii", data=np.array(self.radii))

        self.file.attrs["n_genes"] = self.n_genes
        self.file.attrs["n_radii"] = self.n_radii

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file is not None:
            self.file.close()

    def write_radius(self, radius_idx: int, matrix: np.ndarray) -> None:
        """Write matrix for a specific radius."""
        self.file["morans_i_matrices"][radius_idx, :, :] = matrix.astype(self.dtype)
        self.file.flush()
