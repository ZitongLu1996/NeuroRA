# -*- coding: utf-8 -*-

"""
A module for calculating correlations between two cross-temporal RDMs.
"""

__author__ = "Zitong Lu"

import numpy as np
from scipy.stats import spearmanr


def _validate_ctrdm(ctrdm, name="ctrdm"):
    """
    Validate a cross-temporal representational dissimilarity matrix (CTRDM).

    Parameters
    ----------
    ctrdm : array_like, shape [n_cons, n_cons]
        Cross-temporal RDM.

    name : str, default="ctrdm"
        Variable name used in error messages.

    Returns
    -------
    ctrdm : ndarray, shape [n_cons, n_cons]
        Validated CTRDM.
    """

    ctrdm = np.asarray(ctrdm, dtype=float)

    if ctrdm.ndim != 2:
        raise ValueError(f"{name} should be a 2-D array with shape [n_cons, n_cons].")

    if ctrdm.shape[0] != ctrdm.shape[1]:
        raise ValueError(f"{name} should be a square matrix with shape [n_cons, n_cons].")

    if ctrdm.shape[0] < 2:
        raise ValueError(f"{name} should contain at least 2 conditions.")

    return ctrdm


def _vectorize_ctrdm(ctrdm):
    """
    Vectorize a CTRDM by taking all off-diagonal elements.

    Parameters
    ----------
    ctrdm : ndarray, shape [n_cons, n_cons]
        Cross-temporal RDM.

    Returns
    -------
    vector : ndarray, shape [n_cons * (n_cons - 1)]
        Vectorized off-diagonal CTRDM values.
    """

    n_cons = ctrdm.shape[0]
    offdiag_mask = ~np.eye(n_cons, dtype=bool)

    return ctrdm[offdiag_mask]


def ctrdm_correlation_spearman(ctrdm1, ctrdm2):
    """
    Calculate the Spearman correlation between two CTRDMs.

    Parameters
    ----------
    ctrdm1 : array_like, shape [n_cons, n_cons]
        First cross-temporal RDM.

    ctrdm2 : array_like, shape [n_cons, n_cons]
        Second cross-temporal RDM.

    Returns
    -------
    corr : ndarray, shape [2]
        Spearman correlation result.

        corr[0] is the correlation coefficient.
        corr[1] is the p-value.

    Notes
    -----
    Unlike standard symmetric RDMs, CTRDMs are not assumed to be symmetric.
    Therefore, all off-diagonal elements are used, not just the upper triangle.
    """

    ctrdm1 = _validate_ctrdm(ctrdm1, name="ctrdm1")
    ctrdm2 = _validate_ctrdm(ctrdm2, name="ctrdm2")

    if ctrdm1.shape != ctrdm2.shape:
        raise ValueError("ctrdm1 and ctrdm2 should have the same shape.")

    v1 = _vectorize_ctrdm(ctrdm1)
    v2 = _vectorize_ctrdm(ctrdm2)

    valid_mask = np.isfinite(v1) & np.isfinite(v2)

    if np.sum(valid_mask) < 2:
        return np.array([np.nan, np.nan])

    r, p = spearmanr(v1[valid_mask], v2[valid_mask])

    return np.array([r, p])