"""
Shared statistics helpers for the validation scripts.

bh_fdr() was previously defined in experiment_cascade_validation.py (an
OncoKB-enrichment experiment not reported in the paper, removed 2026-10-03)
and is kept here unchanged. Matches statsmodels multipletests(method="fdr_bh").
"""

import numpy as np


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in input order."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.array(p_values)[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank down
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out.tolist()
