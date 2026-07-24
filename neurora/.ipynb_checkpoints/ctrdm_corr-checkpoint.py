# -*- coding: utf-8 -*-

' a module for calculating the Similarity/Correlation Coefficient between two Cross-temporal RDMs '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from neurora.stuff import permutation_corr


' a function for calculating the Spearman correlation coefficient between two CTRDMs '

def ctrdm_correlation_spearman(CTRDM1, CTRDM2):
    """
    Calculate the similarity based on Spearman Correlation Coefficient between two CTRDMs

    Parameters
    ----------
    CTRDM1 : array [n_conditions, n_conditions]
        The CTRDM 1.
        The shape of CTRDM1 must be [n_cons, n_cons]. n_cons represent the number of conidtions.
    CTRDM2 : array [n_conditions, n_conditions]
        The CTRDM 2.
        The shape of CTRDM2 must be [n_cons, n_cons]. n_cons represent the number of conidtions.


    Returns
    -------
    corr : array [r, p].
        The Spearman Correlation result.
        The shape of corr is [2], including a r-value and a p-value.
    """

    # get number of conditions
    n_cons = np.shape(CTRDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = n_cons * (n_cons - 1)

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n])
    v2 = np.zeros([n])

    # assignment
    nn = 0
    for i in range(n_cons):
        for j in range(n_cons):
            if i != j:
                v1[nn] = CTRDM1[i, j]
                v2[nn] = CTRDM2[i, j]
                nn = nn + 1

    # calculate the Spearman Correlation
    corr = np.array(spearmanr(v1, v2))

    return corr


' a function for calculating the similarity based on Pearson Correlation Coefficient between two CTRDMs '

def ctrdm_correlation_pearson(CTRDM1, CTRDM2):
    """
    Calculate the similarity based on Pearson Correlation Coefficient between two CTRDMs

    Parameters
    ----------
    CTRDM1 : array [n_conditions, n_conditions]
        The CTRDM 1.
        The shape of CTRDM1 must be [n_cons, n_cons]. n_cons represent the number of conidtions.
    CTRDM2 : array [n_conditions, n_conditions]
        The CTRDM 2.
        The shape of CTRDM2 must be [n_cons, n_cons]. n_cons represent the number of conidtions.

    Returns
    -------
    corr : array [r, p].
        The Pearson Correlation result.
        The shape of corr is [2], including a r-value and a p-value.
    """

    # get number of conditions
    n_cons = np.shape(CTRDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = n_cons * (n_cons - 1)

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n])
    v2 = np.zeros([n])

    # assignment
    nn = 0
    for i in range(n_cons):
        for j in range(n_cons):
            if i != j:
                v1[nn] = CTRDM1[i, j]
                v2[nn] = CTRDM2[i, j]
                nn = nn + 1

    # calculate the Pearson Correlation
    corr = np.array(pearsonr(v1, v2))

    return corr


' a function for calculating the similarity based on Kendalls tau Correlation Coefficient between two CTRDMs '

def ctrdm_correlation_kendall(CTRDM1, CTRDM2):
    """
    Calculate the similarity based on Kendalls tau Correlation Coefficient between two CTRDMs

    Parameters
    ----------
    CTRDM1 : array [n_conditions, n_conditions]
        The CTRDM 1.
        The shape of CTRDM1 must be [n_cons, n_cons]. n_cons represent the number of conidtions.
    CTRDM2 : array [n_conditions, n_conditions]
        The CTRDM 2.
        The shape of CTRDM2 must be [n_cons, n_cons]. n_cons represent the number of conidtions.

    Returns
    -------
    corr : array [r, p].
        The Kendalls tau Correlation result.
        The shape of corr is [2], including a r-value and a p-value.
    """

    # get number of conditions
    n_cons = np.shape(CTRDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = n_cons * (n_cons - 1)

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n])
    v2 = np.zeros([n])

    # assignment
    nn = 0
    for i in range(n_cons):
        for j in range(n_cons):
            if i != j:
                v1[nn] = CTRDM1[i, j]
                v2[nn] = CTRDM2[i, j]
                nn = nn + 1

    # calculate the Kendalls tau Correlation
    corr = np.array(kendalltau(v1, v2))

    return corr


def ctrdm_similarity(CTRDM1, CTRDM2):
    """
    Calculate the similarity based on Cosine Similarity between two CTRDMs

    Parameters
    ----------
    CTRDM1 : array [n_conditions, n_conditions]
        The CTRDM 1.
        The shape of CTRDM1 must be [n_cons, n_cons]. n_cons represent the number of conidtions.
    CTRDM2 : array [n_conditions, n_conditions]
        The CTRDM 2.
        The shape of CTRDM2 must be [n_cons, n_cons]. n_cons represent the number of conidtions.

    Returns
    -------
    similarity : float
        The Cosine Similarity result.
    """

    # get number of conditions
    n_cons = np.shape(CTRDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = n_cons * (n_cons - 1)

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n])
    v2 = np.zeros([n])

    # assignment
    nn = 0
    for i in range(n_cons):
        for j in range(n_cons):
            if i != j:
                v1[nn] = CTRDM1[i, j]
                v2[nn] = CTRDM2[i, j]
                nn = nn + 1

    # calculate the Cosine Similarity
    V1 = np.mat(v1)
    V2 = np.mat(v2)
    num = float(V1 * V2.T)
    denom = np.linalg.norm(V1) * np.linalg.norm(V2)
    cos = num / denom
    similarity = 0.5 + 0.5 * cos

    return similarity


' a function for calculating the similarity based on Euclidean Distance between two CTRDMs '

def ctrdm_distance(CTRDM1, CTRDM2):
    """
    Calculate the similarity based on Euclidean Distance between two CTRDMs

    Parameters
    ----------
    CTRDM1 : array [n_conditions, n_conditions]
        The CTRDM 1.
        The shape of CTRDM1 must be [n_cons, n_cons]. n_cons represent the number of conidtions.
    CTRDM2 : array [n_conditions, n_conditions]
        The CTRDM 2.
        The shape of CTRDM2 must be [n_cons, n_cons]. n_cons represent the number of conidtions.

    Returns
    -------
    dist : float.
        The Euclidean Distance result.
    """

    # get number of conditions
    n_cons = np.shape(CTRDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = n_cons * (n_cons - 1)

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n])
    v2 = np.zeros([n])

    # assignment
    nn = 0
    for i in range(n_cons):
        for j in range(n_cons):
            if i != j:
                v1[nn] = CTRDM1[i, j]
                v2[nn] = CTRDM2[i, j]
                nn = nn + 1

    # calculate the Euclidean Distance
    dist = np.linalg.norm(v1 - v2)

    return dist