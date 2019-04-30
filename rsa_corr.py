# -*- coding: utf-8 -*-

' a module for calculating the correlation coefficient between two RDMs '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr

' a function for calculating the Spearman correlation coefficient between two RDMs '
def rsa_correlation_spearman(RDM1, RDM2):

    cons = np.shape(RDM1)[0]

    n = cons * cons

    rdm1 = np.reshape(RDM1, [n])
    rdm2 = np.reshape(RDM2, [n])

    return spearmanr(rdm1, rdm2)

' a function for calculating the Pearson correlation coefficient between two RDMs '
def rsa_correlation_pearson(RDM1, RDM2):

    cons1, cons2 = np.shape(RDM1)

    n = cons1 * cons2

    rdm1 = np.reshape(RDM1, [n])
    rdm2 = np.reshape(RDM2, [n])

    return pearsonr(rdm1, rdm2)

' a function for calculating the Cosine Similarity between two RDMs '

def rsa_similarity(RDM1, RDM2):

    cons1, cons2 = np.shape(RDM1)

    RDM1 = np.reshape(RDM1, [cons1*cons2])
    RDM2 = np.reshape(RDM2, [cons1*cons2])

    rdm1 = np.mat(RDM1)
    rdm2 = np.mat(RDM2)

    num = float(rdm1 * rdm2.T)

    denom = np.linalg.norm(rdm1) * np.linalg.norm(rdm2)

    cos = num / denom

    similarity = 0.5 + 0.5 * cos

    return similarity