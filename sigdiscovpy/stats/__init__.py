"""Statistical testing modules."""

from sigdiscovpy.stats.fdr import apply_fdr_correction
from sigdiscovpy.stats.permutation import permutation_test, batch_permutation_test

__all__ = [
    "apply_fdr_correction",
    "permutation_test",
    "batch_permutation_test",
]
