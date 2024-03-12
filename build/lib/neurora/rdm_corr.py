# -*- coding: utf-8 -*-

' a module for calculating the Similarity/Correlation Coefficient between two RDMs '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from neurora.stuff import permutation_corr


' a function for calculating the Spearman correlation coefficient between two RDMs '

def rdm_correlation_spearman(RDM1, RDM2, rescale=False, permutation=False, iter=1000):

    """
    Calculate the Spearman Correlation between two RDMs

    Parameters
    ----------
    RDM1 : array [ncons, ncons]
        The RDM 1.
        The shape of RDM1 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of RDM2 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    rescale : bool True or False. Default is False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the values on the diagonal.
    permutation : bool True or False. Default is False.
        Conduct permutation test or not.
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    corr : array [r, p].
        The Spearman Correlation result.
        The shape of corr is [2], including a r-value and a p-value.
    """

    if len(np.shape(RDM1)) != 2 or len(np.shape(RDM2)) != 2 or np.shape(RDM1)[0] != np.shape(RDM1)[1] or \
            np.shape(RDM2)[0] != np.shape(RDM2)[1]:

        print("\nThe shapes of two RDMs should be [ncons, ncons]!\n")

        return "Invalid input!"

    # get number of conditions
    cons = np.shape(RDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = int(cons*(cons-1)/2)

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

    # calculate the Spearman Correlation
    rp = np.array(spearmanr(v1, v2))

    if permutation == True:

        rp[1] = permutation_corr(v1, v2, method="spearman", iter=iter)

    return rp


' a function for calculating the Pearson correlation coefficient between two RDMs '

def rdm_correlation_pearson(RDM1, RDM2, rescale=False, permutation=False, iter=1000):

    """
    Calculate the Pearson Correlation between two RDMs

    Parameters
    ----------
    RDM1 : array [ncons, ncons]
        The RDM 1.
        The shape of RDM1 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of RDM2 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    rescale : bool True or False. Default is False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the values on the diagonal.
    permutation : bool True or False. Default is False.
        Conduct permutation test or not.
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    corr : array [r, p].
        The Pearson Correlation result.
        The shape of corr is [2], including a r-value and a p-value.
    """

    if len(np.shape(RDM1)) != 2 or len(np.shape(RDM2)) != 2 or np.shape(RDM1)[0] != np.shape(RDM1)[1] or \
            np.shape(RDM2)[0] != np.shape(RDM2)[1]:

        print("\nThe shapes of two RDMs should be [ncons, ncons]!\n")

        return "Invalid input!"

    # get number of conditions
    cons = np.shape(RDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = int(cons*(cons-1)/2)

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
    v1 = np.zeros([n])
    v2 = np.zeros([n])

    # assignment
    nn = 0
    for i in range(cons - 1):
        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]
            nn = nn + 1

    # calculate the Spearman Correlation
    rp = np.array(pearsonr(v1, v2))

    if permutation == True:

        rp[1] = permutation_corr(v1, v2, method="pearson", iter=iter)

    return rp


' a function for calculating the Kendalls tau correlation coefficient between two RDMs '

def rdm_correlation_kendall(RDM1, RDM2, rescale=False, permutation=False, iter=1000):

    """
    Calculate the Kendalls tau Correlation between two RDMs

    Parameters
    ----------
    RDM1 : array [ncons, ncons]
        The RDM 1.
        The shape of RDM1 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of RDM2 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    rescale : bool True or False. Default is False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the values on the diagonal.
    permutation : bool True or False. Default is False.
        Conduct permutation test or not.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    corr : array [r, p].
        The Kendalls tau Correlation result.
        The shape of corr is [2], including a r-value and a p-value.
    """

    if len(np.shape(RDM1)) != 2 or len(np.shape(RDM2)) != 2 or np.shape(RDM1)[0] != np.shape(RDM1)[1] or \
            np.shape(RDM2)[0] != np.shape(RDM2)[1]:

        print("\nThe shapes of two RDMs should be [ncons, ncons]!\n")

        return "Invalid input!"

    # get number of conditions
    cons = np.shape(RDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = int(cons*(cons-1)/2)

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
    v1 = np.zeros([n])
    v2 = np.zeros([n])

    # assignment
    nn = 0
    for i in range(cons - 1):
        for j in range(cons - 1 - i):
            v1[nn] = RDM1[i, i + j + 1]
            v2[nn] = RDM2[i, i + j + 1]
            nn = nn + 1

    # calculate the Kendalltau Correlation
    rp = np.array(kendalltau(v1, v2))

    if permutation == True:

        rp[1] = permutation_corr(v1, v2, method="kendalltau", iter=iter)

    return rp


' a function for calculating the Cosine Similarity between two RDMs '

def rdm_similarity(RDM1, RDM2, rescale=False):

    """
    Calculate the Cosine Similarity between two RDMs

    Parameters
    ----------
    RDM1 : array [ncons, ncons]
        The RDM 1.
        The shape of RDM1 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of RDM2 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    rescale : bool True or False. Default is False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the values on the diagonal.

    Returns
    -------
    similarity : float.
        The Cosine Similarity result.
    """

    if len(np.shape(RDM1)) != 2 or len(np.shape(RDM2)) != 2 or np.shape(RDM1)[0] != np.shape(RDM1)[1] or \
            np.shape(RDM2)[0] != np.shape(RDM2)[1]:

        print("\nThe shapes of two RDMs should be [ncons, ncons]!\n")

        return "Invalid input!"

    # get number of conditions
    cons = np.shape(RDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = int(cons*(cons-1)/2)

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
    v1 = np.zeros([n])
    v2 = np.zeros([n])

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
    RDM1 : array [ncons, ncons]
        The RDM 1.
        The shape of RDM1 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    RDM2 : array [ncons, ncons].
        The RDM 2.
        The shape of RDM2 must be [n_cons, n_cons].
        n_cons represent the number of conidtions.
    rescale : bool True or False. Default is False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the values on the diagonal.

    Returns
    -------
    dist : float.
        The Euclidean Distance result.
    """

    if len(np.shape(RDM1)) != 2 or len(np.shape(RDM2)) != 2 or np.shape(RDM1)[0] != np.shape(RDM1)[1] or np.shape(RDM2)[0] != np.shape(RDM2)[1]:

        return "Invalid input!"

    # get number of conditions
    cons = np.shape(RDM1)[0]

    # calculate the number of value above the diagonal in RDM
    n = int(cons*(cons-1)/2)

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
    v1 = np.zeros([n])
    v2 = np.zeros([n])

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