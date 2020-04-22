# -*- coding: utf-8 -*-

' a module for calculating the spatiotemporal pattern similarity based on neural data '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import pearsonr
import statsmodels.stats.weightstats as sw
import math

np.seterr(divide='ignore', invalid='ignore')


' a function for calculating the spatiotemporal pattern similarities (STPS) '

def stps(data1, data2, time_win=20, time_step=1):

    """
    calculate the spatiotemporal pattern similarities (STPS)

    Parameters
    ----------
    data1 : array
        The data under condition 1.
        The shape of data1 must be [n_subs1, n_chls, n_trials1, n_ts]. n_subs1, n_chls, n_trials1, n_ts represent the
        number of subjects, the number of channels or regions, the number of trials and the number of time-points of the
        data under condition 1.
    data2 : array
        The data under condition 2.
        The shape of data1 must be [n_subs2, n_chls, n_trials2, n_ts]. n_subs2, n_chls, n_trials2, n_ts represent the
        number of subjects, the number of channels or regions, the number of trials and the number of time-points of the
        data under condition 2.
    time_win : int. Default is 20.
        Set a time-window for calculating the STPS for different time-points.
        If time_win=20, that means each calculation process based on 20 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.

    Returns
    -------
    stps : array
        The STPS.
        The shape of stps is [n_chls, int((n_ts-time_win)/time_step)+1, 2].
        2 representation a t-value and a p-value.
    """

    # for data1

    # get the number of subjects, channels & trials of data1
    subs1, chls, trials1 = np.shape(data1)[:3]

    # the time-points for calculating NPS
    ts = np.shape(data1)[3]
    ts = int((ts - time_win) / time_step) + 1

    # initialize the feature vectors
    feature_vectors1 = np.zeros([subs1, chls, trials1, ts, time_win], dtype=np.float)

    # assigment
    for i in range(subs1):
        for j in range(chls):
            for k in range(trials1):
                for l in range(ts):
                    feature_vectors1[i, j, k, l] = data1[i, j, k, l*time_step:l*time_step+time_win]

    # initialize the correlation matrices
    correlation_matrices1 = np.zeros([chls, ts, trials1, trials1], dtype=np.float)

    # reshape the feature_vectors
    # shape pf feature_vectors: [subs1, chls, trials1, ts, time_win] -> [chls, ts, trials1, subs1, time_win]
    #                           -> [chls, ts, trials1, subs1*time_win]
    feature_vectors1 = np.transpose(feature_vectors1, (1, 3, 2, 0, 4))
    feature_vectors1 = np.reshape(feature_vectors1, [chls, ts, trials1, subs1*time_win])

    # calculate the correlation matrices
    for i in range(chls):
        for j in range(ts):
            for k in range(trials1):
                for l in range(trials1):
                    r = pearsonr(feature_vectors1[i, j, k], feature_vectors1[i, j, l])[0]
                    correlation_matrices1[i, j, k, l] = r

    # calculate the number of values above the diagonal
    n1 = int(trials1*(trials1-1)/2)

    # initialize the correlations1
    correlations1 = np.zeros([chls, ts, n1], dtype=np.float)

    # assignment
    index = 0
    for i in range(chls):
        for j in range(ts):
            for k in range(trials1):
                for l in range(trials1):

                    if l > k:
                        correlations1[index] = correlation_matrices1[i, j, k, l]
                        index = index + 1

    # convert correlations1 to Fisher's Z scores
    correlations1 = 0.5*math.log((1+correlations1)/(1-correlations1))


    # for data2

    # get the number of subjects, channels & trials of data2
    subs2, chls, trials2 = np.shape(data2)[:3]

    # the time-points for calculating NPS
    ts = np.shape(data2)[3]
    ts = int((ts - time_win) / time_step) + 1

    # initialize the feature vectors
    feature_vectors2 = np.zeros([subs2, chls, trials2, ts, time_win], dtype=np.float)

    # assigment
    for i in range(subs2):
        for j in range(chls):
            for k in range(trials2):
                for l in range(ts):
                    feature_vectors2[i, j, k, l] = data2[i, j, k, l * time_step:l * time_step + time_win]

    # initialize the correlation matrices
    correlation_matrices2 = np.zeros([chls, ts, trials2, trials2], dtype=np.float)

    # reshape the feature_vectors
    # shape pf feature_vectors: [subs2, chls, trials2, ts, time_win] -> [chls, ts, trials2, subs2, time_win]
    #                           -> [chls, ts, trials2, subs2*time_win]
    feature_vectors2 = np.transpose(feature_vectors2, (1, 3, 2, 0, 4))
    feature_vectors2 = np.reshape(feature_vectors2, [chls, ts, trials2, subs2 * time_win])

    # calculate the correlation matrices
    for i in range(chls):
        for j in range(ts):
            for k in range(trials2):
                for l in range(trials2):
                    r = pearsonr(feature_vectors2[i, j, k], feature_vectors2[i, j, l])[0]
                    correlation_matrices2[i, j, k, l] = r

    # calculate the number of values above the diagonal
    n2 = int(trials2 * (trials2 - 1) / 2)

    # initialize the correlations2
    correlations2 = np.zeros([chls, ts, n2], dtype=np.float)

    # assignment
    index = 0
    for i in range(chls):
        for j in range(ts):
            for k in range(trials2):
                for l in range(trials2):

                    if l > k:
                        correlations2[index] = correlation_matrices2[i, j, k, l]
                        index = index + 1

    # convert correlations1 to Fisher's Z scores
    correlations2 = 0.5 * math.log((1 + correlations2) / (1 - correlations2))

    # initialize the STPS
    stps = np.zeros([chls, ts, 2])

    # calculate the STPS
    for i in range(chls):
        for j in range(ts):

            # Z-test
            stps[i, j] = sw.ztest(correlations1[i, j], correlations2[i, j])

    return stps
