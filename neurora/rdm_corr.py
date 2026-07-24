# -*- coding: utf-8 -*-

' a module for calculating the Similarity/Spearman Correlation Coefficient between two RDMs '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import spearmanr


def rdm_correlation_spearman(rdm1, rdm2):
    """
    Calculate the Spearman correlation between two RDMs.

    Parameters
    ----------
    rdm1 : array_like, shape [n_cons, n_cons]
        The first RDM.

    rdm2 : array_like, shape [n_cons, n_cons]
        The second RDM.

    Returns
    -------
    corr : ndarray, shape [2]
        Spearman correlation result.

        - corr[0] : Spearman correlation coefficient
        - corr[1] : p-value
    """

    rdm1 = np.asarray(rdm1)
    rdm2 = np.asarray(rdm2)

    if rdm1.ndim != 2 or rdm2.ndim != 2:
        raise ValueError(
            "Both rdm1 and rdm2 should be 2D arrays with shape [n_cons, n_cons]."
        )

    if rdm1.shape[0] != rdm1.shape[1] or rdm2.shape[0] != rdm2.shape[1]:
        raise ValueError(
            "Both rdm1 and rdm2 should be square matrices with shape [n_cons, n_cons]."
        )

    if rdm1.shape != rdm2.shape:
        raise ValueError(
            "rdm1 and rdm2 should have the same shape."
        )

    n_cons = rdm1.shape[0]

    if n_cons < 2:
        raise ValueError(
            "RDMs should contain at least two conditions."
        )

    # Extract the upper-triangular values excluding the diagonal.
    triu_idx = np.triu_indices(n_cons, k=1)
    v1 = rdm1[triu_idx]
    v2 = rdm2[triu_idx]

    r, p = spearmanr(v1, v2)

    return np.array([r, p], dtype=np.float64)