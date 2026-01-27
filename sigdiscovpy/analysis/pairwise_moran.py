"""
Pairwise Moran's I matrix computation.

Computes the gene x gene spatial correlation matrix.
"""

from typing import Optional
import numpy as np
from scipy import sparse as sp_sparse
from sigdiscovpy.gpu.backend import GPU_AVAILABLE, ensure_numpy
from sigdiscovpy.core.normalization import standardize_matrix
from sigdiscovpy.core.weights import create_gaussian_weights, get_weight_sum


def pairwise_moran(
    expr,
    coords,
    radius: float = 100.0,
    sigma: Optional[float] = None,
    W=None,
    normalize: bool = True,
    sparse_W: bool = True,
    same_spot: bool = False,
    use_gpu: bool = True,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """
    Compute pairwise Moran's I matrix across all genes.

    Formula: I = (Z @ W @ Z.T) / weight_sum

    Parameters
    ----------
    expr : array-like
        Expression matrix of shape (n_genes, n_spots) or (n_spots, n_genes).
        Automatically detected based on coords.
    coords : array-like
        Spatial coordinates of shape (n_spots, 2).
    radius : float, default=100.0
        Maximum radius for spatial weights.
    sigma : float, optional
        Gaussian decay parameter. Default: radius / 3.
    W : sparse matrix, optional
        Pre-computed weight matrix. If None, computed from coords.
    normalize : bool, default=True
        Whether to z-score normalize expression before computation.
    sparse_W : bool, default=True
        Use sparse weight matrix for memory efficiency.
    same_spot : bool, default=False
        Include self-connections (diagonal).
    use_gpu : bool, default=True
        Use GPU acceleration if available.
    chunk_size : int, optional
        Gene chunk size for memory management. Auto-computed if None.

    Returns
    -------
    np.ndarray
        Pairwise Moran's I matrix of shape (n_genes, n_genes).

    Examples
    --------
    >>> expr = np.random.randn(100, 1000)  # 100 genes x 1000 spots
    >>> coords = np.random.randn(1000, 2) * 100
    >>> I_matrix = pairwise_moran(expr, coords, radius=50)
    >>> I_matrix.shape
    (100, 100)

    Notes
    -----
    This function matches the sigdiscov R package pairwise_moran() output.
    Uses Gaussian distance decay weights with sigma = radius / 3.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_spots = coords.shape[0]

    # Detect expression matrix orientation
    expr = np.asarray(expr, dtype=np.float64)
    if expr.shape[0] == n_spots:
        # (n_spots, n_genes) format - transpose to (n_genes, n_spots)
        expr = expr.T
    elif expr.shape[1] != n_spots:
        raise ValueError(
            f"Expression shape {expr.shape} incompatible with {n_spots} spots. "
            f"Expected (n_spots, n_genes) or (n_genes, n_spots)."
        )

    n_genes = expr.shape[0]

    # Normalize if requested
    if normalize:
        expr = standardize_matrix(expr, axis=1, use_gpu=use_gpu)

    # Create weight matrix if not provided
    if W is None:
        W = create_gaussian_weights(
            coords,
            radius=radius,
            sigma=sigma,
            same_spot=same_spot,
            sparse=sparse_W,
            row_normalize=False,  # Don't row-normalize for pairwise Moran
            use_gpu=use_gpu,
        )

    weight_sum = get_weight_sum(W)

    if weight_sum < 1e-10:
        return np.zeros((n_genes, n_genes), dtype=np.float64)

    # Compute pairwise Moran's I matrix
    if use_gpu and GPU_AVAILABLE:
        return _pairwise_moran_gpu(expr, W, weight_sum, chunk_size)
    else:
        return _pairwise_moran_cpu(expr, W, weight_sum, chunk_size)


def _pairwise_moran_cpu(
    expr: np.ndarray,
    W,
    weight_sum: float,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """CPU implementation of pairwise Moran's I."""
    # Compute Z @ W: (n_genes x n_spots) @ (n_spots x n_spots) = (n_genes x n_spots)
    if sp_sparse.issparse(W):
        # Sparse: expr @ W.T (since W is symmetric)
        ZW = np.asarray(expr @ W.T)
    else:
        ZW = expr @ W.T

    # Compute (Z @ W) @ Z.T: (n_genes x n_spots) @ (n_spots x n_genes) = (n_genes x n_genes)
    I_matrix = (ZW @ expr.T) / weight_sum

    return I_matrix


def _pairwise_moran_gpu(
    expr: np.ndarray,
    W,
    weight_sum: float,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """GPU implementation of pairwise Moran's I."""
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    n_genes, n_spots = expr.shape

    # Determine chunk size based on GPU memory
    if chunk_size is None:
        try:
            free_mem, _ = cp.cuda.runtime.memGetInfo()
            bytes_per_element = 8  # float64
            # Each gene needs n_spots * 8 bytes, plus intermediate results
            mem_per_gene = n_spots * bytes_per_element * 4
            chunk_size = min(n_genes, max(100, int(free_mem * 0.3 / mem_per_gene)))
        except Exception:
            chunk_size = 1000

    expr_gpu = cp.asarray(expr, dtype=cp.float64)

    if sp_sparse.issparse(W):
        W_gpu = cp_sparse.csr_matrix(W.astype(np.float64))
        ZW_gpu = expr_gpu @ W_gpu.T
    else:
        W_gpu = cp.asarray(W, dtype=cp.float64)
        ZW_gpu = expr_gpu @ W_gpu.T

    I_matrix_gpu = (ZW_gpu @ expr_gpu.T) / weight_sum

    return ensure_numpy(I_matrix_gpu)


def pairwise_moran_directional(
    sender_expr,
    receiver_expr,
    sender_coords,
    receiver_coords,
    radius: float = 100.0,
    sigma: Optional[float] = None,
    normalize: bool = True,
    row_normalize_W: bool = True,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Compute directional pairwise Moran's I for sender -> receiver analysis.

    Parameters
    ----------
    sender_expr : array-like
        Sender expression matrix (n_genes, n_senders).
    receiver_expr : array-like
        Receiver expression matrix (n_genes, n_receivers).
    sender_coords : array-like
        Sender cell coordinates (n_senders, 2).
    receiver_coords : array-like
        Receiver cell coordinates (n_receivers, 2).
    radius : float, default=100.0
        Maximum radius for connections.
    sigma : float, optional
        Gaussian decay parameter.
    normalize : bool, default=True
        Whether to z-score normalize expression.
    row_normalize_W : bool, default=True
        Whether to row-normalize weight matrix.
    use_gpu : bool, default=True
        Use GPU acceleration if available.

    Returns
    -------
    np.ndarray
        Pairwise Moran's I matrix (n_genes, n_genes).
        Entry [i, j] is the spatial correlation of sender gene i
        with receiver gene j in the neighborhood.
    """
    from sigdiscovpy.core.weights import create_directional_weights

    sender_expr = np.asarray(sender_expr, dtype=np.float64)
    receiver_expr = np.asarray(receiver_expr, dtype=np.float64)
    sender_coords = np.asarray(sender_coords, dtype=np.float64)
    receiver_coords = np.asarray(receiver_coords, dtype=np.float64)

    n_senders = sender_coords.shape[0]
    n_receivers = receiver_coords.shape[0]

    # Ensure expression is (n_genes, n_cells)
    if sender_expr.shape[0] == n_senders:
        sender_expr = sender_expr.T
    if receiver_expr.shape[0] == n_receivers:
        receiver_expr = receiver_expr.T

    # Normalize
    if normalize:
        sender_expr = standardize_matrix(sender_expr, axis=1, use_gpu=use_gpu)
        receiver_expr = standardize_matrix(receiver_expr, axis=1, use_gpu=use_gpu)

    # Create directional weight matrix (n_senders, n_receivers)
    W = create_directional_weights(
        sender_coords,
        receiver_coords,
        radius=radius,
        sigma=sigma,
        row_normalize=row_normalize_W,
        use_gpu=use_gpu,
    )

    # Compute spatial lag of receiver expression at sender locations
    # W: (n_senders, n_receivers)
    # receiver_expr.T: (n_receivers, n_genes)
    # lag_G: (n_senders, n_genes) - spatial lag of each gene at sender locations
    if sp_sparse.issparse(W):
        lag_G = np.asarray(W @ receiver_expr.T)
    else:
        lag_G = W @ receiver_expr.T

    # Compute I_ND matrix: cosine similarity between sender expr and spatial lag
    # For pair (f, g): I_ND[f, g] = dot(sender_f, lag_g) / (||sender_f|| * ||lag_g||)
    # sender_expr: (n_genes, n_senders)
    # lag_G: (n_senders, n_genes)

    # sender_expr @ lag_G = (n_genes, n_senders) @ (n_senders, n_genes) = (n_genes, n_genes)
    # This gives dot products for all (f, g) pairs

    # Compute norms
    sender_norms = np.linalg.norm(sender_expr, axis=1)  # (n_genes,)
    lag_norms = np.linalg.norm(lag_G, axis=0)  # (n_genes,)

    # Compute dot products
    dot_products = sender_expr @ lag_G  # (n_genes, n_genes)

    # Normalize to get cosine similarity
    # denominator[i, j] = sender_norms[i] * lag_norms[j]
    denominator = np.outer(sender_norms, lag_norms)
    denominator = np.where(denominator < 1e-10, 1.0, denominator)

    I_matrix = dot_products / denominator

    return I_matrix
