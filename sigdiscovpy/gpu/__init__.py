"""GPU backend and optimization utilities."""

from sigdiscovpy.gpu.backend import get_array_module, GPU_AVAILABLE, ensure_numpy

__all__ = ["get_array_module", "GPU_AVAILABLE", "ensure_numpy"]
