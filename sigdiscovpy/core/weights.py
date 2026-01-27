"""
Spatial weight matrix creation.

Implements:
- Gaussian distance decay weights
- Ring/annular weights for distance-specific effects
- Row normalization

Weight matrix formula:
    w(d) = exp(-(d/sigma)^2 / 2)  where sigma = radius / 3
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy import sparse as sp_sparse
from sigdiscovpy.gpu.backend import GPU_AVAILABLE


def _compute_distances_chunked(
    coords1: np.ndarray,
    coords2: np.ndarray,
    radius: float,
    inner_radius: float = 0.0,
    chunk_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pairwise distances within radius threshold using chunking.

    Returns sparse coordinate format (rows, cols, distances).
    """
    n1 = coords1.shape[0]

    rows_list = []
    cols_list = []
    dists_list = []

    radius_sq = radius * radius
    inner_radius_sq = inner_radius * inner_radius

    for chunk_start in range(0, n1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n1)
        coords1_chunk = coords1[chunk_start:chunk_end]

        # Compute distances: (chunk_size x n2)
        # Use explicit reshape for broadcasting: (chunk_size, 1) - (1, n2)
        diff_x = coords1_chunk[:, 0:1] - coords2[:, 0].reshape(1, -1)
        diff_y = coords1_chunk[:, 1:2] - coords2[:, 1].reshape(1, -1)
        dist_sq = diff_x**2 + diff_y**2

        # Find pairs within radius
        mask = dist_sq <= radius_sq
        if inner_radius > 0:
            mask = mask & (dist_sq > inner_radius_sq)

        rows, cols = np.where(mask)
        rows_list.append(rows + chunk_start)
        cols_list.append(cols)
        dists_list.append(np.sqrt(dist_sq[mask]))

    if rows_list:
        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)
        all_dists = np.concatenate(dists_list)
        return all_rows, all_cols, all_dists
    else:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([])


def create_gaussian_weights(
    coords,
    radius: float,
    sigma: Optional[float] = None,
    same_spot: bool = False,
    sparse: bool = True,
    row_normalize: bool = False,
    use_gpu: bool = True,
) -> Union[np.ndarray, sp_sparse.csr_matrix]:
    """
    Create Gaussian distance decay weight matrix.

    Weight formula: w(d) = exp(-(d/sigma)^2 / 2)

    Parameters
    ----------
    coords : array-like
        Spatial coordinates of shape (n, 2).
    radius : float
        Maximum distance for non-zero weights.
    sigma : float, optional
        Gaussian decay parameter. Default: radius / 3.
    same_spot : bool, default=False
        Whether to include self-connections (diagonal).
    sparse : bool, default=True
        Return sparse matrix for memory efficiency.
    row_normalize : bool, default=False
        Whether to row-normalize the weight matrix.
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    scipy.sparse.csr_matrix or np.ndarray
        Weight matrix of shape (n, n).

    Examples
    --------
    >>> coords = np.random.randn(100, 2) * 100
    >>> W = create_gaussian_weights(coords, radius=50)
    >>> W.shape
    (100, 100)

    Notes
    -----
    Default sigma = radius / 3 means weights drop to ~0.01 at the radius boundary.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]

    if sigma is None:
        sigma = radius / 3.0

    gaussian_factor = -1.0 / (2.0 * sigma * sigma)

    if use_gpu and GPU_AVAILABLE:
        return _create_gaussian_weights_gpu(
            coords, radius, gaussian_factor, same_spot, sparse, row_normalize
        )

    # CPU implementation
    rows, cols, dists = _compute_distances_chunked(coords, coords, radius)

    if len(rows) == 0:
        if sparse:
            return sp_sparse.csr_matrix((n, n), dtype=np.float64)
        return np.zeros((n, n), dtype=np.float64)

    # Remove diagonal if same_spot=False
    if not same_spot:
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]
        dists = dists[mask]

    # Compute Gaussian weights
    weights = np.exp(dists**2 * gaussian_factor)

    # Create sparse matrix
    W = sp_sparse.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float64)

    if row_normalize:
        W = row_normalize_weights(W)

    if not sparse:
        W = W.toarray()

    return W


def _create_gaussian_weights_gpu(
    coords: np.ndarray,
    radius: float,
    gaussian_factor: float,
    same_spot: bool,
    sparse: bool,
    row_normalize: bool,
):
    """GPU implementation of Gaussian weight matrix creation.

    Uses try-finally to ensure GPU memory cleanup on exceptions.
    """
    import cupy as cp

    n = coords.shape[0]
    coords_gpu = None
    rows_list = []
    cols_list = []
    weights_list = []

    try:
        coords_gpu = cp.asarray(coords, dtype=cp.float32)

        # Estimate chunk size based on GPU memory
        try:
            free_mem, _ = cp.cuda.runtime.memGetInfo()
            bytes_per_element = 4  # float32
            mem_per_row = n * bytes_per_element * 4  # 4 arrays
            chunk_size = min(n, int(free_mem * 0.3 / mem_per_row))
            chunk_size = max(chunk_size, 100)
        except Exception:
            chunk_size = 5000

        radius_sq = radius * radius

        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            coords_chunk = coords_gpu[chunk_start:chunk_end]

            # Compute distances: (chunk_size, 1) - (1, n) -> (chunk_size, n)
            # Use explicit reshape for proper broadcasting
            diff_x = coords_chunk[:, 0:1] - coords_gpu[:, 0].reshape(1, -1)
            diff_y = coords_chunk[:, 1:2] - coords_gpu[:, 1].reshape(1, -1)
            dist_sq = diff_x * diff_x + diff_y * diff_y

            # Mask within radius
            mask = dist_sq <= radius_sq
            if not same_spot:
                diag_mask = cp.arange(chunk_start, chunk_end).reshape(-1, 1) == cp.arange(n)
                mask = mask & ~diag_mask

            # Compute weights
            weights_chunk = cp.exp(dist_sq * gaussian_factor) * mask

            # Extract non-zero entries
            nonzero_mask = weights_chunk > 1e-10
            if cp.any(nonzero_mask):
                rows, cols = cp.where(nonzero_mask)
                rows_list.append(rows + chunk_start)
                cols_list.append(cols)
                weights_list.append(weights_chunk[nonzero_mask])

            # Explicit cleanup of chunk arrays
            del diff_x, diff_y, dist_sq, mask, weights_chunk, nonzero_mask

        if not rows_list:
            if sparse:
                return sp_sparse.csr_matrix((n, n), dtype=np.float64)
            return np.zeros((n, n), dtype=np.float64)

        # Concatenate and convert to CPU
        all_rows = cp.concatenate(rows_list).get()
        all_cols = cp.concatenate(cols_list).get()
        all_weights = cp.concatenate(weights_list).get().astype(np.float64)

        W = sp_sparse.csr_matrix(
            (all_weights, (all_rows, all_cols)), shape=(n, n), dtype=np.float64
        )

        if row_normalize:
            W = row_normalize_weights(W)

        if not sparse:
            W = W.toarray()

        return W

    finally:
        # Ensure GPU memory is freed even on exception
        if coords_gpu is not None:
            del coords_gpu
        # Clear lists to release GPU array references
        for arr in rows_list:
            del arr
        for arr in cols_list:
            del arr
        for arr in weights_list:
            del arr
        rows_list.clear()
        cols_list.clear()
        weights_list.clear()
        # Force synchronization to ensure memory is released
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass


def create_ring_weights(
    coords,
    outer_radius: float,
    inner_radius: float,
    sigma: Optional[float] = None,
    sparse: bool = True,
    row_normalize: bool = False,
    use_gpu: bool = True,
) -> Union[np.ndarray, sp_sparse.csr_matrix]:
    """
    Create annular (ring) weight matrix for distance-specific effects.

    Only includes connections between inner_radius < d <= outer_radius.

    Parameters
    ----------
    coords : array-like
        Spatial coordinates of shape (n, 2).
    outer_radius : float
        Outer edge of the annulus.
    inner_radius : float
        Inner edge of the annulus.
    sigma : float, optional
        Gaussian decay parameter. Default: outer_radius / 3.
    sparse : bool, default=True
        Return sparse matrix.
    row_normalize : bool, default=False
        Whether to row-normalize.
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    scipy.sparse.csr_matrix or np.ndarray
        Weight matrix with annular structure.

    Examples
    --------
    >>> coords = np.random.randn(100, 2) * 100
    >>> W = create_ring_weights(coords, outer_radius=100, inner_radius=50)
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]

    if sigma is None:
        sigma = outer_radius / 3.0

    gaussian_factor = -1.0 / (2.0 * sigma * sigma)

    # Compute distances within outer radius, excluding inner radius
    rows, cols, dists = _compute_distances_chunked(
        coords, coords, outer_radius, inner_radius=inner_radius
    )

    if len(rows) == 0:
        if sparse:
            return sp_sparse.csr_matrix((n, n), dtype=np.float64)
        return np.zeros((n, n), dtype=np.float64)

    # Compute Gaussian weights
    weights = np.exp(dists**2 * gaussian_factor)

    W = sp_sparse.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float64)

    if row_normalize:
        W = row_normalize_weights(W)

    if not sparse:
        W = W.toarray()

    return W


def row_normalize_weights(W) -> sp_sparse.csr_matrix:
    """
    Row-normalize a weight matrix.

    Each row sums to 1 (for non-zero rows). Rows with zero sum (isolated cells
    with no neighbors) remain unchanged (all zeros).

    Parameters
    ----------
    W : sparse matrix or np.ndarray
        Weight matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-normalized weight matrix.

    Examples
    --------
    >>> W = sp_sparse.random(100, 100, density=0.1)
    >>> W_norm = row_normalize_weights(W)
    >>> np.allclose(W_norm.sum(axis=1), 1, atol=1e-10)
    True

    Notes
    -----
    **Isolated Cell Handling (Zero-Sum Rows)**:

    When a row has zero sum (cell has no neighbors within radius), the row
    remains all zeros after normalization. This means:

    1. The spatial lag for that cell will be 0: ``lag[i] = W[i, :] @ z = 0``
    2. This is the correct behavior - isolated cells have no neighbor information
    3. Metrics (Moran's I, I_ND) will treat these cells as having no spatial signal

    If you need to identify isolated cells, check for zero row sums before
    normalization:

    >>> row_sums = np.array(W.sum(axis=1)).ravel()
    >>> isolated_cells = np.where(row_sums == 0)[0]

    **Matching R Implementation**:

    This behavior matches the sigdiscov R package, where isolated cells (no
    neighbors within the specified radius) produce zero spatial lag values.
    """
    if not sp_sparse.issparse(W):
        W = sp_sparse.csr_matrix(W)
    else:
        W = W.tocsr()

    row_sums = np.array(W.sum(axis=1)).ravel()
    row_sums_inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)

    # Create diagonal matrix and multiply
    D_inv = sp_sparse.diags(row_sums_inv, format="csr")
    W_normalized = D_inv @ W

    return W_normalized


def create_directional_weights(
    sender_coords,
    receiver_coords,
    radius: float,
    sigma: Optional[float] = None,
    row_normalize: bool = True,
    use_gpu: bool = True,
) -> sp_sparse.csr_matrix:
    """
    Create directional weight matrix for sender -> receiver analysis.

    Parameters
    ----------
    sender_coords : array-like
        Coordinates of sender cells (n_senders x 2).
    receiver_coords : array-like
        Coordinates of receiver cells (n_receivers x 2).
    radius : float
        Maximum distance for connections.
    sigma : float, optional
        Gaussian decay parameter.
    row_normalize : bool, default=True
        Whether to row-normalize.
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    scipy.sparse.csr_matrix
        Weight matrix of shape (n_senders, n_receivers).
    """
    sender_coords = np.asarray(sender_coords, dtype=np.float64)
    receiver_coords = np.asarray(receiver_coords, dtype=np.float64)

    n_senders = sender_coords.shape[0]
    n_receivers = receiver_coords.shape[0]

    if sigma is None:
        sigma = radius / 3.0

    gaussian_factor = -1.0 / (2.0 * sigma * sigma)

    rows, cols, dists = _compute_distances_chunked(sender_coords, receiver_coords, radius)

    if len(rows) == 0:
        return sp_sparse.csr_matrix((n_senders, n_receivers), dtype=np.float64)

    weights = np.exp(dists**2 * gaussian_factor)

    W = sp_sparse.csr_matrix(
        (weights, (rows, cols)), shape=(n_senders, n_receivers), dtype=np.float64
    )

    if row_normalize:
        W = row_normalize_weights(W)

    return W


def get_weight_sum(W) -> float:
    """Get the total sum of weights in a matrix."""
    if sp_sparse.issparse(W):
        return float(W.sum())
    return float(np.sum(W))
