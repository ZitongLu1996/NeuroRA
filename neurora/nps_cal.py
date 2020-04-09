# -*- coding: utf-8 -*-

' a module for calculating the neural pattern similarity based neural data '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import pearsonr

np.seterr(divide='ignore', invalid='ignore')

' a function for calculating the neural pattern similarity '

def nps(data, time_win=5, sub_opt=0):
    # shape of data: [2, n_subs, n_trials, n_chls, n_ts]
    #                 here 2 means calculating the nps between the neural activities under two conditions
    # time_win: the time-window, if time_win=5, that means each calculation process bases on 5 time points
    #           this is also a processing of downsampling
    # sub_opt : 0: consider all the subjects
    #           1: calculating independently for each subject

    nsubs, ntrials, nchls, nts = data.shape[1:]

    avgdata = np.average(data, axis=2)

    ts = int(nts/time_win)

    if sub_opt == 1:

        nps = np.zeros([nsubs, nchls, ts])

        for sub in range(nsubs):

            for i in range(nchls):
                for j in range(ts):

                    data1 = avgdata[0, sub, i, j*time_win:j*time_win+time_win]
                    data2 = avgdata[1, sub, i, j*time_win:j*time_win+time_win]

                    nps[sub, i, j] = pearsonr(data1, data2)[0]

        return nps

    # if sub_opt == 0

    nps = np.zeros([nchls, ts])

    for i in range(nchls):
        for j in range(ts):

            data1 = avgdata[0, :, i, j*time_win:j*time_win+time_win]
            data2 = avgdata[1, :, i, j*time_win:j*time_win+time_win]

            data1 = np.reshape(data1, nsubs*time_win)
            data2 = np.reshape(data2, nsubs*time_win)

            nps[i, j] = pearsonr(data1, data2)[0]

    return nps