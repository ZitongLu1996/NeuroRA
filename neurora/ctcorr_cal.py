# -*- coding: utf-8 -*-

' a module for calculating the cross-temporal similarity '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import spearmanr
from neurora.rdm_corr import rdm_correlation_spearman, rdm_correlation_pearson, rdm_correlation_kendall, \
    rdm_similarity, rdm_distance
from neurora.ctrdm_corr import ctrdm_correlation_spearman, ctrdm_correlation_pearson, ctrdm_correlation_kendall, \
    ctrdm_similarity, ctrdm_distance
from neurora.stuff import show_progressbar


' a function for calculating Cross-Temporal Similarities between neural data under two conditions '

def ctsim_cal(data1, data2, sub_opt=1, chl_opt=0, time_win=5, time_step=5):

    """
    Calculate the Cross-Temporal Similarities between neural data under two conditions

    Parameters
    ----------
    data1 : array
        EEG/MEG/fNIRS/EEG-like data from a time-window under condition1.
        The shape of data1 must be [n_subs, n_chls, n_ts]. n_subs, n_chls, n_ts represent the number of conditions, the
        number of subjects, the number of channels and the number of time-points respectively.
    data2 : array
        EEG/MEG/fNIRS/EEG-like data from a time-window under condition2.
        The shape of data2 must be [n_subs, n_chls, n_ts]. n_subs, n_chls, n_ts represent the number of conditions, the
        number of subjects, the number of channels and the number of time-points respectively.
    sub_opt : int 0 or 1. Default is 1.
        Caculate the similarities for each subject or not.
        If sub_opt=0, calculating based on all data.
        If sub_opt=1, calculating based on each subject's data, respectively.
    chl_opt : int 0 or 1. Default is 0.
        Caculate the similarities for each channel or not.
        If chl_opt=0, calculating based on all channels' data.
        If chl_opt=1, calculating based on each channel's data respectively.
    time_win : int. Default is 5.
        Set a time-window for calculating the similarities for different time-points.
        If time_win=5, that means each calculation process based on 10 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.

    Returns
    -------
    CTSimilarities : array
        Cross-temporal similarities.
        If sub_opt=1 and chl_opt=1, the shape of CTSimilarities will be [n_subs, n_channels,
        int((n_ts-time_win)/time_step)+1, int((n_ts-time_win)/time_step)+1, 2]. If sub_opt=1 and chl_opt=0, the shape of
        CTSimilarities will be [n_subs, int((n_ts-time_win)/time_step)+1, int((n_ts-time_win)/time_step)+1, 2]. If
        sub_opt=0 and chl_opt=1, the shape of CTSimilarities will be [n_channels, int((n_ts-time_win)/time_step)+1,
        int((n_ts-time_win)/time_step)+1, 2]. If sub_opt=0 and chl_opt=0, the shape of CTSimilarities will be
        [int((n_ts-time_win)/time_step)+1, int((n_ts-time_win)/time_step)+1, 2]
    """

    n_subs, n_chls, n_ts = np.shape(data1)

    nts = int((n_ts - time_win) / time_step) + 1

    # chl_opt=0
    if chl_opt == 0:

        newdata1 = np.zeros([n_subs, nts, n_chls, time_win])
        newdata2 = np.zeros([n_subs, nts, n_chls, time_win])

        for sub in range(n_subs):
            for t in range(nts):
                for chl in range(n_chls):
                    newdata1[sub, t, chl] = data1[sub, chl, t*time_step:t*time_step+time_win]
                    newdata2[sub, t, chl] = data2[sub, chl, t*time_step:t*time_step+time_win]

        newdata1 = np.reshape(newdata1, [n_subs, nts, n_chls*time_win])
        newdata2 = np.reshape(newdata2, [n_subs, nts, n_chls*time_win])

        CTSimilarities = np.zeros([n_subs, nts, nts, 2])

        for sub in range(n_subs):
            for t1 in range(nts):
                for t2 in range(nts):

                    CTSimilarities[sub, t1, t2] = spearmanr(newdata1[sub, t1], newdata2[sub, t2])

        if sub_opt == 0:

            CTSimilarities = np.average(CTSimilarities, axis=0)

            return CTSimilarities

        if sub_opt == 1:

            return CTSimilarities

    if chl_opt == 1:

        newdata1 = np.zeros([n_subs, n_chls, nts, time_win])
        newdata2 = np.zeros([n_subs, n_chls, nts, time_win])

        for sub in range(n_subs):
            for chl in range(n_chls):
                for t in range(nts):
                    newdata1[sub, chl, t] = data1[sub, chl, t * time_step:t * time_step + time_win]
                    newdata2[sub, chl, t] = data2[sub, chl, t * time_step:t * time_step + time_win]

        CTSimilarities = np.zeros([n_subs, n_chls, nts, nts, 2])

        for sub in range(n_subs):
            for chl in range(n_chls):
                for t1 in range(nts):
                    for t2 in range(nts):

                        CTSimilarities[sub, chl, t1, t2] = spearmanr(newdata1[sub, chl, t1], newdata2[sub, chl, t2])

        if sub_opt == 0:

            CTSimilarities = np.average(CTSimilarities, axis=0)

            return CTSimilarities

        if sub_opt == 1:

            return CTSimilarities


' a function for calculating Cross-Temporal Similarity Matrix based on temporal RDMs '

def ctsim_rdms_cal(RDMs, method='spearman'):

    """
    Calculate the Cross-Temporal Similarities based on RDMs in time series

    Parameters
    ----------
    RDMs : array
        The Representational Dissimilarity Matrices in time series.
        The shape could be [n_ts, n_cons, n_cons] or [n_subs, n_ts, n_cons, n_cons] or [n_chls, n_ts, n_cons, n_cons] or
        [n_subs, n_chls, n_ts, n_cons, n_cons]. n_ts, n_conditions, n_subs, n_chls represent the number of time-points,
        the number of conditions, the number of subjects and the number of channels, respectively.
    method : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or 'distance'. Default is 'spearman'.
        The method to calculate the similarities.
        If method='spearman', calculate the Spearman Correlations. If method='pearson', calculate the Pearson
        Correlations. If methd='kendall', calculate the Kendall tau Correlations. If method='similarity', calculate the
        Cosine Similarities. If method='distance', calculate the Euclidean Distances.

    Returns
    -------
    CTSimilarities : array
        Cross-temporal similarities.
        If the shape of RDMs is [n_ts, n_cons, n_cons] and method='spearman' or 'pearson' or 'kendall', the shape of
        CTSimilarities will be [n_ts, n_ts, 2].
        If the shape of RDMs is [n_subs, n_ts, n_cons, n_cons] and method='spearman' or 'pearson' or 'kendall', the
        shape of CTSimilarities will be [n_subs, n_ts, n_ts, 2].
        If the shape of RDMs is [n_chls, n_ts, n_cons, n_cons] and method='spearman' or 'pearson' or 'kendall', the
        shape of CTSimilarities will be [n_chls, n_ts, n_ts, 2].
        If the shape of RDMs is [n_subs, n_channels, n_ts, n_cons, n_cons] and method='spearman' or 'pearson' or
        'kendall', the shape of CTSimilarities will be [n_subs, n_channels, n_ts, n_ts, 2].
        If the shape of RDMs is [n_ts, n_cons, n_cons] and method='similarity' or 'distance', the shape of
        CTSimilarities will be [n_ts, n_ts].
        If the shape of RDMs is [n_subs, n_ts, n_cons, n_cons] and method='similarity' or 'distance', the shape of
        CTSimilarities will be [n_subs, n_ts, n_ts].
        If the shape of RDMs is [n_chls, n_ts, n_cons, n_cons] and method='similarity' or 'distance', the shape of
        CTSimilarities will be [n_chls, n_ts, n_ts].
        If the shape of RDMs is [n_subs, n_channels, n_ts, n_cons, n_cons] and method='similarity' or 'distance', the
        shape of CTSimilarities will be [n_subs, n_chls, n_ts, n_ts].
    """

    n = len(np.shape(RDMs))

    if n == 3:

        n_ts, n_cons = np.shape(RDMs)[:2]

        CTSimilarities = np.zeros([n_ts, n_ts, 2])

        for t1 in range(n_ts):
            for t2 in range(n_ts):

                if method == 'spearman':
                    CTSimilarities[t1, t2] = rdm_correlation_spearman(RDMs[t1], RDMs[t2])
                if method == 'pearson':
                    CTSimilarities[t1, t2] = rdm_correlation_pearson(RDMs[t1], RDMs[t2])
                if method == 'kendall':
                    CTSimilarities[t1, t2] = rdm_correlation_kendall(RDMs[t1], RDMs[t2])
                if method == 'similarity':
                    CTSimilarities[t1, t2, 0] = rdm_similarity(RDMs[t1], RDMs[t2])
                if method == 'distance':
                    CTSimilarities[t1, t2, 0] = rdm_distance(RDMs[t1], RDMs[t2])

        if method == 'spearman' or method == 'pearson' or method == 'kendall':

            return CTSimilarities

        if method == 'similarity' or method == 'distance':

            return CTSimilarities[:, :, 0]

    if n == 4:

        n1, n_ts, n_cons = np.shape(RDMs)[:3]

        CTSimilarities = np.zeros([n1, n_ts, n_ts, 2])

        for i in range(n1):
            for t1 in range(n_ts):
                for t2 in range(n_ts):

                    if method == 'spearman':
                        CTSimilarities[i, t1, t2] = rdm_correlation_spearman(RDMs[i, t1], RDMs[i, t2])
                    if method == 'pearson':
                        CTSimilarities[i, t1, t2] = rdm_correlation_pearson(RDMs[i, t1], RDMs[i, t2])
                    if method == 'kendall':
                        CTSimilarities[i, t1, t2] = rdm_correlation_kendall(RDMs[i, t1], RDMs[i, t2])
                    if method == 'similarity':
                        CTSimilarities[i, t1, t2, 0] = rdm_similarity(RDMs[i, t1], RDMs[i, t2])
                    if method == 'distance':
                        CTSimilarities[i, t1, t2, 0] = rdm_distance(RDMs[i, t1], RDMs[i, t2])

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, 0]

    if n == 5:

        n1, n2, n_ts, n_cons = np.shape(RDMs)[:4]

        CTSimilarities = np.zeros([n1, n2, n_ts, n_ts, 2])

        for i in range(n1):
            for j in range(n2):
                for t1 in range(n_ts):
                    for t2 in range(n_ts):

                        if method == 'spearman':
                            CTSimilarities[i, j, t1, t2] = rdm_correlation_spearman(RDMs[i, j, t1], RDMs[i, j, t2])
                        if method == 'pearson':
                            CTSimilarities[i, j, t1, t2] = rdm_correlation_pearson(RDMs[i, j, t1], RDMs[i, j, t2])
                        if method == 'kendall':
                            CTSimilarities[i, j, t1, t2] = rdm_correlation_kendall(RDMs[i, j, t1], RDMs[i, j, t2])
                        if method == 'similarity':
                            CTSimilarities[i, j, t1, t2, 0] = rdm_similarity(RDMs[i, j, t1], RDMs[i, j, t2])
                        if method == 'distance':
                            CTSimilarities[i, j, t1, t2, 0] = rdm_distance(RDMs[i, j, t1], RDMs[i, j, t2])

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, :, 0]


' a function for calculating Cross-Temporal Similarities between CTRDMs and a Coding Model RDM '

def ctsim_ctrdms_cal(CTRDMs, Model_RDM, method='spearman'):

    """
    Calculate the Cross-Temporal Similarities between CTRDMs and a Coding Model RDM

    Parameters
    ----------
    CTRDMs : array
        The Cross-Temporal Representational Dissimilarity Matrices.
        The shape could be [n_ts, n_ts, n_cons, n_cons] or [n_subs, n_ts, n_ts, n_cons, n_cons] or [n_chls, n_ts,
        n_ts, n_cons, n_cons] or [n_subs, n_chls, n_ts, n_ts, n_cons, n_cons]. n_ts, n_cons, n_subs, n_chls represent
        the number of time-points, the number of conditions, the number of subjects and the number of channels,
        respectively.
    Model_RDM : array [n_cons, n_cons].
        The Coding Model RDM.
    method : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or 'distance'. Default is 'spearman'.
        The method to calculate the similarities.
        If method='spearman', calculate the Spearman Correlations. If method='pearson', calculate the Pearson
        Correlations. If methd='kendall', calculate the Kendall tau Correlations. If method='similarity', calculate the
        Cosine Similarities. If method='distance', calculate the Euclidean Distances.

    Returns
    -------
    CTSimilarities : array
        Cross-temporal similarities.
        If method='spearman' or 'pearson' or 'kendall':
            If the shape of CTRDMs is [n_ts, n_ts, n_cons, n_cons], the shape of CTSimilarities will be [n_ts, n_ts, 2].
            If the shape of CTRDMs is [n_subs, n_ts, n_ts, n_cons, n_cons], the shape of CTSimilarities will be [n_subs,
            n_ts, n_ts, 2].
            If the shape of CTRDMs is [n_channels, n_ts, n_ts, n_cons, n_cons], the shape of CTSimilarities will be
            [n_channels, n_ts, n_ts, 2].
            If the shape of CTRDMs is [n_subs, n_channels, n_ts, n_ts, n_cons, n_cons], the shape of CTSimilarities will
            be [n_subs, n_channels, n_ts, n_ts, 2].
        If method='similarity' or 'distance':
            If the shape of CTRDMs is [n_ts, n_ts, n_cons, n_cons], the shape of CTSimilarities will be [n_ts, n_ts].
            If the shape of CTRDMs is [n_subs, n_ts, n_ts, n_cons, n_cons], the shape of CTSimilarities will be [n_subs,
            n_ts, n_ts].
            If the shape of CTRDMs is [n_channels, n_ts, n_ts, n_cons, n_cons], the shape of CTSimilarities will be
            [n_channels, n_ts, n_ts].
            If the shape of CTRDMs is [n_subs, n_channels, n_ts, n_ts, n_cons, n_cons], the shape of CTSimilarities will
            be [n_subs, n_channels, n_ts, n_ts].
    """

    n = len(np.shape(CTRDMs))

    if n == 4:

        n_ts, n_cons = np.shape(CTRDMs)[1:3]

        CTSimilarities = np.zeros([n_ts, n_ts, 2])

        total = n_ts * n_ts

        for t1 in range(n_ts):
            for t2 in range(n_ts):

                percent = (t1 * n_ts + t2) / total * 100
                show_progressbar("Calculating", percent)

                if method == 'spearman':
                    CTSimilarities[t1, t2] = ctrdm_correlation_spearman(CTRDMs[t1, t2], Model_RDM)
                if method == 'pearson':
                    CTSimilarities[t1, t2] = ctrdm_correlation_pearson(CTRDMs[t1, t2], Model_RDM)
                if method == 'kendall':
                    CTSimilarities[t1, t2] = ctrdm_correlation_kendall(CTRDMs[t1, t2], Model_RDM)
                if method == 'similarity':
                    CTSimilarities[t1, t2, 0] = ctrdm_similarity(CTRDMs[t1, t2], Model_RDM)
                if method == 'distance':
                    CTSimilarities[t1, t2, 0] = ctrdm_distance(CTRDMs[t1, t2], Model_RDM)

        if method == 'spearman' or method == 'pearson' or method == 'kendall':

            return CTSimilarities

        if method == 'similarity' or method == 'distance':

            return CTSimilarities[:, :, 0]

    if n == 5:

        n1 = np.shape(CTRDMs)[0]
        n_ts, n_cons = np.shape(CTRDMs)[2:4]

        CTSimilarities = np.zeros([n1, n_ts, n_ts, 2])

        total = n1 * n_ts * n_ts

        for i in range(n1):
            for t1 in range(n_ts):
                for t2 in range(n_ts):

                    percent = (i * n_ts * n_ts + t1 * n_ts + t2) / total * 100
                    show_progressbar("Calculating", percent)

                    if method == 'spearman':
                        CTSimilarities[i, t1, t2] = ctrdm_correlation_spearman(CTRDMs[i, t1, t2], Model_RDM)
                        #print(CTSimilarities[i, t1, t2])
                    if method == 'pearson':
                        CTSimilarities[i, t1, t2] = ctrdm_correlation_pearson(CTRDMs[i, t1, t2], Model_RDM)
                    if method == 'kendall':
                        CTSimilarities[i, t1, t2] = ctrdm_correlation_kendall(CTRDMs[i, t1, t2], Model_RDM)
                    if method == 'similarity':
                        CTSimilarities[i, t1, t2, 0] = ctrdm_similarity(CTRDMs[i, t1, t2], Model_RDM)
                    if method == 'distance':
                        CTSimilarities[i, t1, t2, 0] = ctrdm_distance(CTRDMs[i, t1, t2], Model_RDM)

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, 0]

    if n == 6:

        n1, n2 = np.shape(CTRDMs)[:2]
        n_ts, n_cons = np.shape(CTRDMs)[3:5]

        CTSimilarities = np.zeros([n1, n2, n_ts, n_ts, 2])

        total = n1 * n2 * n_ts * n_ts

        for i in range(n1):
            for j in range(n2):
                for t1 in range(n_ts):
                    for t2 in range(n_ts):

                        percent = (i * n2 * n_ts * n_ts + j * n_ts * n_ts + t1 * n_ts + t2) / total * 100
                        show_progressbar("Calculating", percent)

                        if method == 'spearman':
                            CTSimilarities[i, j, t1, t2] = ctrdm_correlation_spearman(CTRDMs[i, j, t1, t2], Model_RDM)
                        if method == 'pearson':
                            CTSimilarities[i, j, t1, t2] = ctrdm_correlation_pearson(CTRDMs[i, j, t1, t2], Model_RDM)
                        if method == 'kendall':
                            CTSimilarities[i, j, t1, t2] = ctrdm_correlation_kendall(CTRDMs[i, j, t1, t2], Model_RDM)
                        if method == 'similarity':
                            CTSimilarities[i, j, t1, t2, 0] = ctrdm_similarity(CTRDMs[i, j, t1, t2], Model_RDM)
                        if method == 'distance':
                            CTSimilarities[i, j, t1, t2, 0] = ctrdm_distance(CTRDMs[i, j, t1, t2], Model_RDM)

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, :, 0]


' a function for calculating Cross-Temporal Similarities between temporal model RDMs and temporal neural RDMs (dynamic-RSA) '

def ctsim_drsa_cal(Model_RDMs, RDMs, method='spearman'):

    """
    Calculate the Cross-Temporal Similarities between temporal model RDMs and temporal neural RDMs (dynamic-RSA)

    Parameters
    ----------
    Model_RDMs : array
        The Coding Model RDMs.
        The shape should be [n_ts, n_cons, n_cons]. n_ts and n_cons represent the number of time-points and the number
        of conditions, respectively.
    RDMs : array
        The Representational Dissimilarity Matrices in time series.
        The shape could be [n_ts, n_cons, n_cons] or [n_subs, n_ts, n_cons, n_cons] or [n_channels, n_ts, n_cons,
        n_cons] or [n_subs, n_channels, n_ts, n_cons, n_cons]. n_ts, n_cons, n_subs, n_channels represent the number of
        time-points, the number of conditions, the number of subjects and the number of channels, respectively.
    method : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or 'distance'. Default is 'spearman'.
        The method to calculate the similarities.
        If method='spearman', calculate the Spearman Correlations. If method='pearson', calculate the Pearson
        Correlations. If methd='kendall', calculate the Kendall tau Correlations. If method='similarity', calculate the
        Cosine Similarities. If method='distance', calculate the Euclidean Distances.

    Returns
    -------
    CTSimilarities : array
        Cross-temporal similarities.
        If the shape of RDMs is [n_ts, n_cons, n_cons] and method='spearman' or 'pearson' or 'kendall', the shape of
        CTSimilarities will be [n_ts, n_ts, 2].
        If the shape of RDMs is [n_subs, n_ts, n_cons, n_cons] and method='spearman' or 'pearson' or 'kendall', the
        shape of CTSimilarities will be [n_subs, n_ts, n_ts, 2].
        If the shape of RDMs is [n_chls, n_ts, n_cons, n_cons] and method='spearman' or 'pearson' or 'kendall', the
        shape of CTSimilarities will be [n_chls, n_ts, n_ts, 2].
        If the shape of RDMs is [n_subs, n_chls, n_ts, n_cons, n_cons] and method='spearman' or 'pearson' or 'kendall',
        the shape of CTSimilarities will be [n_subs, n_chls, n_ts, n_ts, 2].
        If the shape of RDMs is [n_ts, n_cons, n_cons] and method='similarity' or 'distance', the shape of
        CTSimilarities will be [n_ts, n_ts].
        If the shape of RDMs is [n_subs, n_ts, n_cons, n_cons] and method='similarity' or 'distance', the shape of
        CTSimilarities will be [n_subs, n_ts, n_ts].
        If the shape of RDMs is [n_chls, n_ts, n_cons, n_cons] and method='similarity' or 'distance', the shape of
        CTSimilarities will be [n_chls, n_ts, n_ts].
        If the shape of RDMs is [n_subs, n_chls, n_ts, n_cons, n_cons] and method='similarity' or 'distance', the shape
        of CTSimilarities will be [n_subs, n_chls, n_ts, n_ts].
    """

    n = len(np.shape(RDMs))

    if n == 3:

        n_ts, n_cons = np.shape(RDMs)[:2]

        CTSimilarities = np.zeros([n_ts, n_ts, 2])

        for t1 in range(n_ts):
            for t2 in range(n_ts):

                if method == 'spearman':
                    CTSimilarities[t1, t2] = rdm_correlation_spearman(RDMs[t1], Model_RDMs[t2])
                if method == 'pearson':
                    CTSimilarities[t1, t2] = rdm_correlation_pearson(RDMs[t1], Model_RDMs[t2])
                if method == 'kendall':
                    CTSimilarities[t1, t2] = rdm_correlation_kendall(RDMs[t1], Model_RDMs[t2])
                if method == 'similarity':
                    CTSimilarities[t1, t2, 0] = rdm_similarity(RDMs[t1], Model_RDMs[t2])
                if method == 'distance':
                    CTSimilarities[t1, t2, 0] = rdm_distance(RDMs[t1], Model_RDMs[t2])

        if method == 'spearman' or method == 'pearson' or method == 'kendall':

            return CTSimilarities

        if method == 'similarity' or method == 'distance':

            return CTSimilarities[:, :, 0]

    if n == 4:

        n1, n_ts, n_cons = np.shape(RDMs)[:3]

        CTSimilarities = np.zeros([n1, n_ts, n_ts, 2])

        for i in range(n1):
            for t1 in range(n_ts):
                for t2 in range(n_ts):

                    if method == 'spearman':
                        CTSimilarities[i, t1, t2] = rdm_correlation_spearman(RDMs[i, t1], Model_RDMs[i, t2])
                    if method == 'pearson':
                        CTSimilarities[i, t1, t2] = rdm_correlation_pearson(RDMs[i, t1], Model_RDMs[i, t2])
                    if method == 'kendall':
                        CTSimilarities[i, t1, t2] = rdm_correlation_kendall(RDMs[i, t1], Model_RDMs[i, t2])
                    if method == 'similarity':
                        CTSimilarities[i, t1, t2, 0] = rdm_similarity(RDMs[i, t1], Model_RDMs[i, t2])
                    if method == 'distance':
                        CTSimilarities[i, t1, t2, 0] = rdm_distance(RDMs[i, t1], Model_RDMs[i, t2])

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, 0]

    if n == 5:

        n1, n2, n_ts, n_cons = np.shape(RDMs)[:4]

        CTSimilarities = np.zeros([n1, n2, n_ts, n_ts, 2])

        for i in range(n1):
            for j in range(n2):
                for t1 in range(n_ts):
                    for t2 in range(n_ts):

                        if method == 'spearman':
                            CTSimilarities[i, j, t1, t2] = rdm_correlation_spearman(RDMs[i, j, t1], Model_RDMs[i, j, t2])
                        if method == 'pearson':
                            CTSimilarities[i, j, t1, t2] = rdm_correlation_pearson(RDMs[i, j, t1], Model_RDMs[i, j, t2])
                        if method == 'kendall':
                            CTSimilarities[i, j, t1, t2] = rdm_correlation_kendall(RDMs[i, j, t1], Model_RDMs[i, j, t2])
                        if method == 'similarity':
                            CTSimilarities[i, j, t1, t2, 0] = rdm_similarity(RDMs[i, j, t1], Model_RDMs[i, j, t2])
                        if method == 'distance':
                            CTSimilarities[i, j, t1, t2, 0] = rdm_distance(RDMs[i, j, t1], Model_RDMs[i, j, t2])

        if method == 'spearman' or method == 'pearson' or method == 'kendall':
            return CTSimilarities

        if method == 'similarity' or method == 'distance':
            return CTSimilarities[:, :, :, :, 0]