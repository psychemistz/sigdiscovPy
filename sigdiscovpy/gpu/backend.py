"""
GPU/CPU backend abstraction layer.

Provides a unified interface for NumPy and CuPy operations,
allowing seamless switching between CPU and GPU computation.
"""

import numpy as np

# Try to import CuPy
try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse

    GPU_AVAILABLE = True
except ImportError:
    cp = None
    cp_sparse = None
    GPU_AVAILABLE = False


def get_array_module(use_gpu: bool = True):
    """
    Get the appropriate array module (NumPy or CuPy).

    Parameters
    ----------
    use_gpu : bool, default=True
        Whether to use GPU if available. If False or GPU not available,
        returns NumPy.

    Returns
    -------
    module
        Either cupy or numpy module.

    Examples
    --------
    >>> xp = get_array_module(use_gpu=True)
    >>> x = xp.array([1, 2, 3])
    >>> xp.sum(x)
    6
    """
    if use_gpu and GPU_AVAILABLE:
        return cp
    return np


def get_sparse_module(use_gpu: bool = True):
    """
    Get the appropriate sparse matrix module.

    Parameters
    ----------
    use_gpu : bool, default=True
        Whether to use GPU sparse if available.

    Returns
    -------
    module
        Either cupyx.scipy.sparse or scipy.sparse module.
    """
    if use_gpu and GPU_AVAILABLE:
        return cp_sparse
    from scipy import sparse

    return sparse


def ensure_numpy(arr) -> np.ndarray:
    """
    Convert array to NumPy array (from CuPy if necessary).

    Parameters
    ----------
    arr : array-like
        Input array (NumPy, CuPy, or other array-like).

    Returns
    -------
    np.ndarray
        NumPy array.

    Examples
    --------
    >>> xp = get_array_module(use_gpu=True)
    >>> x_gpu = xp.array([1, 2, 3])
    >>> x_cpu = ensure_numpy(x_gpu)
    >>> type(x_cpu)
    <class 'numpy.ndarray'>
    """
    if GPU_AVAILABLE and hasattr(arr, "get"):
        # CuPy array - transfer to CPU
        return arr.get()
    return np.asarray(arr)


def ensure_gpu(arr, dtype=None):
    """
    Convert array to CuPy array (from NumPy if necessary).

    Parameters
    ----------
    arr : array-like
        Input array.
    dtype : dtype, optional
        Data type for the output array.

    Returns
    -------
    cupy.ndarray or numpy.ndarray
        CuPy array if GPU available, otherwise NumPy array.
    """
    if not GPU_AVAILABLE:
        if dtype is not None:
            return np.asarray(arr, dtype=dtype)
        return np.asarray(arr)

    if hasattr(arr, "__cuda_array_interface__"):
        # Already a CuPy array
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    if dtype is not None:
        return cp.asarray(arr, dtype=dtype)
    return cp.asarray(arr)


def is_gpu_array(arr) -> bool:
    """
    Check if array is a CuPy (GPU) array.

    Parameters
    ----------
    arr : array-like
        Input array.

    Returns
    -------
    bool
        True if arr is a CuPy array.
    """
    if not GPU_AVAILABLE:
        return False
    return hasattr(arr, "__cuda_array_interface__")


def get_gpu_memory_info() -> dict:
    """
    Get GPU memory information.

    Returns
    -------
    dict
        Dictionary with 'free', 'total', 'used' memory in bytes.
        Returns None values if GPU not available.
    """
    if not GPU_AVAILABLE:
        return {"free": None, "total": None, "used": None}

    try:
        free, total = cp.cuda.runtime.memGetInfo()
        return {"free": free, "total": total, "used": total - free}
    except Exception:
        return {"free": None, "total": None, "used": None}


def clear_gpu_memory():
    """Clear GPU memory pool."""
    if GPU_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
