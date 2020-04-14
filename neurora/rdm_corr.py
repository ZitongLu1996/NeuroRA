# -*- coding: utf-8 -*-

' a module for calculating the correlation coefficient between two RDMs '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau


' a function for calculating the Spearman correlation coefficient between two RDMs '

def rdm_correlation_spearman(RDM1, RDM2, rescale=False):

    """
    Calculate the Spearman Correlation between two RDMs

    Parameters
    ----------
    RDM1 : array [ncons, ncons].
        The RDM 1.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    rescale : Boolean True or False.
        Rescale the values in RDM or not. Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagnal.

    Returns
    -------
    corr : array [r, p].
        The Spearman Correlation result.
        The shape of corr is [2], including a r-value and a p-value.
    """

    # get number of conditions
    cons = np.shape(RDM1)[0]
    print(cons)

    # calculate the number of value above the diagonal in RDM
    n = 0
    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    print(np.shape(RDM1))
    print(np.shape(RDM2))

    if rescale == True:

        # flatten the RDM1
        vrdm = np.reshape(RDM1, [cons*cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        # flatten the RDM2
        vrdm = np.reshape(RDM2, [cons * cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    # assignment
    nn = 0
    for i in range(cons-1):
        for j in range(cons-1-i):
            v1[nn] = RDM1[i, i+j+1]
            v2[nn] = RDM2[i, i+j+1]
            nn = nn + 1

    print(v1)
    print(v2)

    # calculate the Spearman Correlation
    return spearmanr(v1, v2)


' a function for calculating the Pearson correlation coefficient between two RDMs '

def rdm_correlation_pearson(RDM1, RDM2, rescale=False):

    """
    Calculate the Pearson Correlation between two RDMs

    Parameters
    ----------
    RDM1 : array [ncons, ncons].
        The RDM 1.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    rescale : Boolean True or False.
        Rescale the values in RDM or not. Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagnal.

    Returns
    -------
    corr : array [r, p].
        The Pearson Correlation result.
        The shape of corr is [2], including a r-value and a p-value.
    """

    # get number of conditions
    cons = np.shape(RDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = 0
    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    if rescale == True:

        # flatten the RDM1
        vrdm = np.reshape(RDM1, [cons*cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        # flatten the RDM2
        vrdm = np.reshape(RDM2, [cons * cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    # assignment
    nn = 0
    for i in range(cons - 1):
        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]
            nn = nn + 1

    # calculate the Pearson Correlation
    return pearsonr(v1, v2)


' a function for calculating the Kendalls tau correlation coefficient between two RDMs '

def rdm_correlation_kendall(RDM1, RDM2, rescale=False):

    """
    Calculate the Kendalls tau Correlation between two RDMs

    Parameters
    ----------
    RDM1 : array [ncons, ncons].
        The RDM 1.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    rescale : Boolean True or False.
        Rescale the values in RDM or not. Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagnal.

    Returns
    -------
    corr : array [r, p].
        The Kendalls tau Correlation result.
        The shape of corr is [2], including a r-value and a p-value.
    """

    # get number of conditions
    cons = np.shape(RDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = 0
    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    if rescale == True:

        # flatten the RDM1
        vrdm = np.reshape(RDM1, [cons*cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        # flatten the RDM2
        vrdm = np.reshape(RDM2, [cons * cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    # assignment
    nn = 0
    for i in range(cons - 1):
        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]
            nn = nn + 1

    # calculate the Kendall tau Correlation
    return kendalltau(v1, v2)


' a function for calculating the Cosine Similarity between two RDMs '

def rdm_similarity(RDM1, RDM2, rescale=False):

    """
    Calculate the Cosine Similarity between two RDMs

    Parameters
    ----------
    RDM1 : array [ncons, ncons].
        The RDM 1.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    rescale : Boolean True or False.
        Rescale the values in RDM or not. Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagnal.

    Returns
    -------
    corr : float
        The Cosine Similarity result.
    """

    # get number of conditions
    cons = np.shape(RDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = 0
    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    if rescale == True:

        # flatten the RDM1
        vrdm = np.reshape(RDM1, [cons*cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        # flatten the RDM2
        vrdm = np.reshape(RDM2, [cons * cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    # assignment
    nn = 0
    for i in range(cons - 1):
        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]
            nn = nn + 1

    # calculate the Cosine Similarity
    V1 = np.mat(v1)
    V2 = np.mat(v2)
    num = float(V1 * V2.T)
    denom = np.linalg.norm(V1) * np.linalg.norm(V2)
    cos = num / denom
    similarity = 0.5 + 0.5 * cos

    return similarity


' a fuction for calculating the Euclidean Distance between two RDMs '

def rdm_distance(RDM1, RDM2, rescale=False):

    """
    Calculate the Euclidean Distance between two RDMs

    Parameters
    ----------
    RDM1 : array [ncons, ncons].
        The RDM 1.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    rescale : Boolean True or False.
        Rescale the values in RDM or not. Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagnal.

    Returns
    -------
    corr : float
        The Euclidean Distance result.
    """

    # get number of conditions
    cons = np.shape(RDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = 0
    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    if rescale == True:

        # flatten the RDM1
        vrdm = np.reshape(RDM1, [cons*cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        # flatten the RDM2
        vrdm = np.reshape(RDM2, [cons * cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    # assignment
    nn = 0
    for i in range(cons - 1):
        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]
            nn = nn + 1

    # calculate the Euclidean Distance
    dist = np.linalg.norm(v1 - v2)

    return dist


' a function for permutation test between two RDMs '

def rdm_permutation(RDM1, RDM2, iter=1000, rescale=False):

    """
    Conduct Permutation test between two RDMs

    Parameters
    ----------
    RDM1 : array [ncons, ncons].
        The RDM 1.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of fmri_data must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    iter : int. Default is 1000.
        The times for iteration.
    rescale : Boolean True or False.
        Rescale the values in RDM or not. Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagnal.

    Returns
    -------
    p : float
        The permutation test result, p-value.
    """

    # get number of conditions
    cons = np.shape(RDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = 0
    while cons > 1:
        n = n + cons - 1
        cons = cons - 1

    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    cons = np.shape(RDM1)[0]

    if rescale == True:

        # flatten the RDM1
        vrdm = np.reshape(RDM1, [cons*cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM1[i, j] = float((RDM1[i, j] - minvalue) / (maxvalue - minvalue))

        # flatten the RDM2
        vrdm = np.reshape(RDM2, [cons * cons])
        # array -> set -> list
        svrdm = set(vrdm)
        lvrdm = list(svrdm)
        lvrdm.sort()

        # get max & min
        maxvalue = lvrdm[-1]
        minvalue = lvrdm[1]

        # rescale
        if maxvalue != minvalue:

            for i in range(cons):
                for j in range(cons):

                    # not on the diagnal
                    if i != j:
                        RDM2[i, j] = float((RDM2[i, j] - minvalue) / (maxvalue - minvalue))

    # initialize two vectors to store the values above the diagnal of two RDMs
    v1 = np.zeros([n], dtype=np.float64)
    v2 = np.zeros([n], dtype=np.float64)

    # assignment
    nn = 0
    for i in range(cons - 1):
        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]
            nn = nn + 1

    # permutation test

    diff = abs(np.average(v1) - np.average(v2))
    v = np.hstack((v1, v2))
    print(v)
    nv = v.shape[0]
    ni = 0

    for i in range(iter):
        vshuffle = np.random.permutation(v)
        vshuffle1 = vshuffle[:int(nv/2)]
        vshuffle2 = vshuffle[int(nv/2):]
        diff_i = abs(np.average(vshuffle1) - np.average(vshuffle2))

        if diff_i >= diff:
            ni = ni + 1

    # permunitation test p-value
    p = ni/iter

    return p
