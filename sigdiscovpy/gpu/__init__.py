"""GPU backend and optimization utilities."""

from sigdiscovpy.gpu.backend import GPU_AVAILABLE, ensure_numpy, get_array_module

__all__ = ["get_array_module", "GPU_AVAILABLE", "ensure_numpy"]
