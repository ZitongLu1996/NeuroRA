# -*- coding: utf-8 -*-

' a module for calculating the RDM based on multimode neural data '

__author__ = 'Zitong Lu'

import numpy as np
from neurora.stuff import limtozero
import math
from scipy.stats import pearsonr

np.seterr(divide='ignore', invalid='ignore')


' a function for calculating the RDM(s) based on behavioral data '

def bhvRDM(bhv_data, sub_opt=0, abs=True):

    """
    Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for behavioral data

    Parameters
    ----------
    bhv_data : array
        The behavioral data.
        The shape of bhv_data must be [n_cons, n_subs, n_trials].
        n_cons, n_subs & n_trials represent the number of conidtions, the number of subjects & the number of trials,
        respectively.
    sub_opt : int 0 or 1. Default is 0.
        Calculate the RDM for each subject or not.
        If sub_opt=0, return only one RDM based on all data.
        If sub_opt=1, return n_subs RDMs based on each subject's data.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not.

    Returns
    -------
    RDM(s) : array
        The behavioral RDM.
        If sub_opt=0, return only one RDM. The shape is [n_cons, n_cons].
        If sub_opt=1, return n_subs RDMs. The shape is [n_subs, n_cons, n_cons].

    Notes
    -----
    This function can also be used to calculate the RDM for computational simulation data
    """

    # get the number of conditions & the number of subjects
    cons = len(bhv_data)

    # get the number of conditions
    n_subs = []

    for i in range(cons):
        n_subs.append(np.shape(bhv_data[i])[0])

    subs = n_subs[0]

    # shape of bhv_data: [N_cons, N_subs, N_trials]

    # save the number of trials of each condition
    n_trials = []

    for i in range(cons):
        n_trials.append(np.shape(bhv_data[i])[1])

    # save the number of trials of each condition
    if len(set(n_trials)) != 1:
            return None

    # sub_opt=1

    if sub_opt == 1:

        # initialize the RDMs
        rdms = np.zeros([subs, cons, cons], dtype=np.float64)

        # calculate the values in RDMs
        for sub in range(subs):
            rdm = np.zeros([cons, cons], dtype=np.float)
            for i in range(cons):
                for j in range(cons):
                    # calculate the difference
                    rdm[i, j] = np.abs(np.average(bhv_data[i, sub])-np.average(bhv_data[j, sub]))

            # flatten the RDM
            vrdm = np.reshape(rdm, [cons * cons])
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
                            rdm[i, j] = float((rdm[i, j] - minvalue) / (maxvalue - minvalue))
            rdms[sub] = rdm

        return rdms

    # & sub_opt=0

    # initialize the RDM
    rdm = np.zeros([cons, cons], dtype=np.float64)

    # judge whether numbers of trials of different conditions are same
    if len(set(n_subs)) != 1:
        return None

    # initialize the data for calculating the RDM
    data = np.zeros([cons, subs], dtype=np.float64)

    # assignment
    for i in range(cons):
        for j in range(subs):
            # save the data for each subject under each condition, average the trials
            data[i][j] = np.average(bhv_data[i][j])

    # calculate the values in RDM
    for i in range(cons):
        for j in range(cons):
            # calculate the Pearson Coefficient
            r = pearsonr(data[i], data[j])[0]
            # calculate the dissimilarity
            if abs == True:
                rdm[i, j] = limtozero(1 - np.abs(r))
            else:
                rdm[i, j] = limtozero(1 - r)

    return rdm


' a function for calculating the RDM(s) based on EEG/MEG/fNIRS data '

def eegRDM(EEG_data, sub_opt=0, chl_opt=0, time_opt=0, time_win=5, time_step=5, abs=True):

    """
    Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for EEG/MEG/fNIRS data

    Parameters
    ----------
    EEG_data : array
        The EEG/MEG/fNIRS data.
        The shape of EEGdata must be [n_cons, n_subs, n_trials, n_chls, n_ts].
        n_cons, n_subs, n_trials, n_chls & n_ts represent the number of conidtions, the number of subjects, the number
        of trials, the number of channels & the number of time-points, respectively.
    sub_opt : int 0 or 1. Default is 0.
        Calculate the RDM for each subject or not.
        If sub_opt=0, return only one RDM based on all data.
        If sub_opt=1, return n_subs RDMs based on each subject's data
    chl_opt : int 0 or 1. Default is 0.
        Calculate the RDM for each channel or not.
        If chl_opt=0, calculate the RDM based on all channels'data.
        If chl_opt=1, calculate the RDMs based on each channel's data respectively.
    time_opt : int 0 or 1. Default is 0.
        Calculate the RDM for each time-point or not
        If time_opt=0, calculate the RDM based on whole time-points' data.
        If time_opt=1, calculate the RDMs based on each time-points respectively.
    time_win : int. Default is 5.
        Set a time-window for calculating the RDM for different time-points.
        Only when time_opt=1, time_win works.
        If time_win=5, that means each calculation process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.
        Only when time_opt=1, time_step works.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not.

    Returns
    -------
    RDM(s) : array
        The EEG/MEG/fNIR RDM.
        If sub_opt=0 & chl_opt=0 & time_opt=0, return only one RDM.
            The shape is [n_cons, n_cons].
        If sub_opt=0 & chl_opt=0 & time_opt=1, return int((n_ts-time_win)/time_step)+1 RDM.
            The shape is [int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
        If sub_opt=0 & chl_opt=1 & time_opt=0, return n_chls RDM.
            The shape is [n_chls, n_cons, n_cons].
        If sub_opt=0 & chl_opt=1 & time_opt=1, return n_chls*(int((n_ts-time_win)/time_step)+1) RDM.
            The shape is [n_chls, int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
        If sub_opt=1 & chl_opt=0 & time_opt=0, return n_subs RDM.
            The shape is [n_subs, n_cons, n_cons].
        If sub_opt=1 & chl_opt=0 & time_opt=1, return n_subs*(int((n_ts-time_win)/time_step)+1) RDM.
            The shape is [n_subs, int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
        If sub_opt=1 & chl_opt=1 & time_opt=0, return n_subs*n_chls RDM.
            The shape is [n_subs, n_chls, n_cons, n_cons].
        If sub_opt=1 & chl_opt=1 & time_opt=1, return n_subs*n_chls*(int((n_ts-time_win)/time_step)+1) RDM.
            The shape is [n_subs, n_chls, int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
    """

    # get the number of conditions, subjects, trials, channels and time points
    cons, subs, trials, chls, ts = np.shape(EEG_data)

    if sub_opt == 1:

        if time_opt == 1:

            # the time-points for calculating RDM
            ts = int((ts-time_win)/time_step)+1

            if chl_opt == 1:

                # sub_opt=1 & time_opt=1 & chl_opt=1

                # initialize the data for calculating the RDM
                data = np.zeros([subs, chls, ts, cons, time_win], dtype=np.float64)

                # assignment
                for i in range(subs):
                    for j in range(chls):
                        for k in range(ts):
                            for l in range(cons):
                                for m in range(time_win):
                                    # average the trials
                                    data[i, j, k, l, m] = np.average(EEG_data[l, i, :, j, k * time_step + m])

                # initialize the RDMs
                rdms = np.zeros([subs, chls, ts, cons, cons], dtype=np.float64)

                # calculate the values in RDMs
                for i in range(subs):
                    for j in range(chls):
                        for k in range(ts):
                            for l in range(cons):
                                for m in range(cons):
                                    # calculate the Pearson Coefficient
                                    r = pearsonr(data[i, j, k, l], data[i, j, k, m])[0]
                                    # calculate the dissimilarity
                                    if abs == True:
                                        rdms[i, j, k, l, m] = limtozero(1 - np.abs(r))
                                    else:
                                        rdms[i, j, k, l, m] = limtozero(1 - r)

                return rdms

            # sub_opt=1 & time_opt=1 & chl_opt=0

            # initialize the data for calculating the RDM
            data = np.zeros([subs, ts, cons, chls, time_win], dtype=np.float64)

            # assignment
            for i in range(subs):
                for j in range(ts):
                    for k in range(cons):
                        for l in range(chls):
                            for m in range(time_win):
                                # average the trials
                                data[i, j, k, l, m] = np.average(EEG_data[k, i, :, l, j * time_step + m])

            # flatten the data for different calculating conditions
            data = np.reshape(data, [subs, ts, cons, chls * time_win])

            # initialize the RDMs
            rdms = np.zeros([subs, ts, cons, cons], dtype=np.float64)

            # calculate the values in RDMs
            for i in range(subs):
                for j in range(ts):
                    for k in range(cons):
                        for l in range(cons):
                            # calculate the Pearson Coefficient
                            r = pearsonr(data[i, j, k], data[i, j, l])[0]
                            # calculate the dissimilarity
                            if abs == True:
                                rdms[i, j, k, l] = limtozero(1 - np.abs(r))
                            else:
                                rdms[i, j, k, l] = limtozero(1 - r)

            return rdms

        # if time_opt = 0

        cons, subs, trials, chls, ts = np.shape(EEG_data)

        if chl_opt == 1:

            # sub_opt=1 & time_opt=0 & chl_opt=1

            # initialize the data for calculating the RDM
            data = np.zeros([subs, chls, cons, ts], dtype=np.float64)

            # assignment
            for i in range(subs):
                for j in range(chls):
                    for k in range(cons):
                        for l in range(ts):
                            # average the trials
                            data[i, j, k, l] = np.average(EEG_data[k, i, :, j, l])

            # initialize the RDMs
            rdms = np.zeros([subs, chls, cons, cons], dtype=np.float64)

            # calculate the values in RDMs
            for i in range(subs):
                for j in range(chls):
                    for k in range(cons):
                        for l in range(cons):
                            # calculate the Pearson Coefficient
                            r = pearsonr(data[i, j, k], data[i, j, l])[0]
                            # calculate the dissimilarity
                            if abs == True:
                                rdms[i, j, k, l] = limtozero(1 - np.abs(r))
                            else:
                                rdms[i, j, k, l] = limtozero(1 - r)

            return rdms

        # if chl_opt = 0

        # sub_opt=1 & time_opt=0 & chl_opt=0

        # initialize the data for calculating the RDM
        data = np.zeros([subs, cons, chls, ts], dtype=np.float64)

        # assignment
        for i in range(subs):
            for j in range(cons):
                for k in range(chls):
                    for l in range(ts):
                        # average the trials
                        data[i, j, k, l] = np.average(EEG_data[j, i, :, k, l])

        # flatten the data for different calculating conditions
        data = np.reshape(data, [subs, cons, chls * ts])

        # initialize the RDMs
        rdms = np.zeros([subs, cons, cons], dtype=np.float64)

        # calculate the values in RDMs
        for i in range(subs):
            for j in range(cons):
                for k in range(cons):
                    # calculate the Pearson Coefficient
                    r = pearsonr(data[i, j], data[i, k])[0]
                    # calculate the dissimilarity
                    if abs == True:
                        rdms[i, j, k] = limtozero(1 - np.abs(r))
                    else:
                        rdms[i, j, k] = limtozero(1 - r)

        return rdms

    # if sub_opt = 0

    if time_opt == 1:

        # the time-points for calculating RDM
        ts = int((ts-time_win)/time_step)+1

        if chl_opt == 1:

            # sub_opt=0 & time_opt=1 & chl_opt=1

            # initialize the data for calculating the RDM
            data = np.zeros([chls, ts, cons, time_win], dtype=np.float64)

            # assignment
            for i in range(chls):
                for j in range(ts):
                    for k in range(cons):
                        for l in range(time_win):
                            # average the trials & subs
                            data[i, j, k, l] = np.average(EEG_data[k, :, :, i, j * time_step + l])

            # initialize the RDMs
            rdms = np.zeros([chls, ts, cons, cons], dtype=np.float64)

            # calculate the values in RDMs
            for i in range(chls):
                for j in range(ts):
                    for k in range(cons):
                        for l in range(cons):
                            # calculate the Pearson Coefficient
                            r = pearsonr(data[i, j, k], data[i, j, l])[0]
                            # calculate the dissimilarity
                            if abs == True:
                                rdms[i, j, k, l] = limtozero(1 - np.abs(r))
                            else:
                                rdms[i, j, k, l] = limtozero(1 - r)

            return rdms

        # sub_opt=0 & time_opt=1 & chl_opt=0

        # initialize the data for calculating the RDM
        data = np.zeros([ts, cons, chls, time_win], dtype=np.float64)

        # assignment
        for i in range(ts):
            for j in range(cons):
                for k in range(chls):
                    for l in range(time_win):
                        # average the subjects & trials
                        data[i, j, k, l] = np.average(EEG_data[j, :, :, k, i * time_step + l])

        # flatten the data for different calculating conditions
        data = np.reshape(data, [ts, cons, chls * time_win])

        # initialize the RDMs
        rdms = np.zeros([ts, cons, cons], dtype=np.float64)

        # calculate the values in RDMs
        for i in range(ts):
            for j in range(cons):
                for k in range(cons):
                    # calculate the Pearson Coefficient
                    r = pearsonr(data[i, j], data[i, k])[0]
                    # calculate the dissimilarity
                    if abs == True:
                        rdms[i, j, k] = limtozero(1 - np.abs(r))
                    else:
                        rdms[i, j, k] = limtozero(1 - r)

        return rdms

    # if time_opt = 0

    # sub_opt=0 & time_opt=0 & chl_opt=1

    # get the number of conditions, subjects, trials, channels & time-points
    cons, subs, trials, chls, ts = np.shape(EEG_data)

    if chl_opt == 1:

        # initialize the data for calculating the RDM
        data = np.zeros([chls, cons, ts], dtype=np.float64)

        # assignment
        for i in range(chls):
            for j in range(cons):
                for k in range(ts):
                    # average the subjects & trials
                    data[i, j, k] = np.average(EEG_data[j, :, :, i, k])

        # flatten the data for different calculating conditions
        data = np.reshape(data, [chls, cons, ts])

        # initialize the RDMs
        rdms = np.zeros([chls, cons, cons], dtype=np.float64)

        # calculate the values in RDMs
        for i in range(chls):
            for j in range(cons):
                for k in range(cons):
                    # calculate the Pearson Coefficient
                    r = pearsonr(data[i, j], data[i, k])[0]
                    # calculate the dissimilarity
                    if abs == True:
                        rdms[i, j, k] = limtozero(1 - np.abs(r))
                    else:
                        rdms[i, j, k] = limtozero(1 - r)

        return rdms

    # if chl_opt = 0

    # sub_opt=0 & time_opt=0 & chl_opt=0

    # get the number of conditions, subjects, trials, channels & time-points
    cons, subs, trials, chls, ts = np.shape(EEG_data)

    # initialize the data for calculating the RDM
    data = np.zeros([cons, chls, ts], dtype=np.float64)

    # assignment
    for i in range(cons):
        for j in range(chls):
            for k in range(ts):
                    # average the subjects & trials
                    data[i, j, k] = np.average(EEG_data[i, :, :, j, k])

    # flatten the data for different calculating condition
    data = np.reshape(data, [cons, chls * ts])

    # initialize the RDM
    rdm = np.zeros([cons, cons], dtype=np.float64)

    # calculate the values in RDM
    for i in range(cons):
        for j in range(cons):
            # calculate the Pearson Coefficient
            r = pearsonr(data[i], data[j])[0]
            # calculate the dissimilarity
            if abs == True:
                rdm[i, j] = limtozero(1 - np.abs(r))
            else:
                rdm[i, j] = limtozero(1 - r)

    return rdm


' a function for calculating the RDM(s) based on ECoG/sEEG/electrophysiological data '

def ecogRDM(ele_data, opt="all", time_win=5, time_step=5, abs=True):

    """
    Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for ECoG/sEEG/electrophysiology data

    Parameters
    ----------
    ele_data : array
        The ECoG/sEEG/electrophysiology data.
        The shape of EEGdata must be [n_cons, n_trials, n_chls, n_ts].
        n_cons, n_trials, n_chls & n_ts represent the number of conidtions, the number of trials,
        the number of channels & the number of time-points, respectively.
    opt : string 'channel', 'time' or 'all'. Default is 'all'.
        Calculate the RDM for each channel or for each time-point or for the whole data.
        If opt='channel', return n_chls RDMs based on each channel's data.
        If opt='time', return int(ts/time_win) RDMs based on each time-point's data respectively.
        If opt='allin', return only one RDM based on all data.
    time_win : int. Default is 5.
        Set a time-window for calculating the RDM for different time-points or not.
        Only when opt='time', time_win works.
        If time_win=5, that means each calculation process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.
        Only when opt='time', time_step works.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not.

    Returns
    -------
    RDM(s) : array
        The ECoG/sEEG/electrophysiology RDM.
        If opt='channel', return n_chls RDM.
            The shape is [n_chls, n_cons, n_cons].
        If opt='time', return int((n_ts-time_win)/time_step)+1 RDM.
            The shape is [int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
        If opt='all', return one RDM.
            The shape is [n_cons, n_cons].
    """

    # get the number of conditins, trials, channels and time points
    cons, trials, chls, ts = np.shape(ele_data)

    # calculate for each channel
    if opt == "channel":

        # initialize the data for calculating the RDM
        data = np.zeros([chls, cons, ts], dtype=np.float64)

        # assignment
        for i in range(chls):
            for j in range(cons):
                for k in range(ts):
                    # average the trials
                    data[i, j, k] = np.average(ele_data[j, :, i, k])

        # initialize the RDMs
        rdms = np.zeros([chls, cons, cons], dtype=np.float64)

        # calculate the values in RDMs
        for i in range(chls):
            for j in range(cons):
                for k in range(cons):
                    # calculate the Pearson Coefficient
                    r = pearsonr(data[i, j], data[i, k])[0]
                    # calculate the dissimilarity
                    if abs == True:
                        rdms[i, j, k] = limtozero(1 - np.abs(r))
                    else:
                        rdms[i, j, k] = limtozero(1 - r)

        return rdms

    # calculate for each time-point
    elif opt == "time":

        # the time-points for calculating RDM
        ts = int((ts-time_win)/time_step)+1

        # initialize the data for calculating the RDM
        data = np.zeros([ts, cons, chls, ts], dtype=np.float64)

        # assignment
        for i in range(ts):
            for j in range(cons):
                for k in range(chls):
                    for l in range(ts):
                        # average the trials
                        data[i, j, k, l] = np.average(ele_data[j, :, k, i*time_step+l])

        # flatten the data for different calculating conditions
        data = np.reshape(data, [ts, cons, chls*ts])

        # initialize the RDMs
        rdms = np.zeros([ts, cons, cons], dtype=np.float64)

        # calculate the values in RDMs
        for i in range(ts):
            for j in range(cons):
                for k in range(cons):
                    # calculate the Pearson Coefficient
                    r = pearsonr(np.sort(data[i, j]), np.sort(data[i, k]))[0]
                    # calculate the dissimilarity
                    if abs == True:
                        rdms[i, j, k] = limtozero(1 - np.abs(r))
                    else:
                        rdms[i, j, k] = limtozero(1 - r)

        return rdms


' a function for calculating the RDM based on fMRI data (searchlight) '

def fmriRDM(fmri_data, ksize=[3, 3, 3], strides=[1, 1, 1], sub_result=0, abs=True):

    """
    Calculate the Representational Dissimilarity Matrices (RDMs) for fMRI data (Searchlight)

    Parameters
    ----------
    fmri_data : array
        The fmri data.
        The shape of fmri_data must be [n_cons, n_subs, nx, ny, nz]. n_cons, nx, ny, nz represent the number of
        conditions, the number of subs & the size of fMRI-img, respectively.
    ksize : array or list [kx, ky, kz]. Default is [3, 3, 3].
        The size of the fMRI-img. nx, ny, nz represent the number of voxels along the x, y, z axis.
    strides : array or list [sx, sy, sz]. Default is [1, 1, 1].
        The strides for calculating along the x, y, z axis.
    sub_result: int 0 or 1. Default is 0.
        Return the subject-result or average-result.
        If sub_result=0, return the average result.
        If sub_result=1, return the results of each subject.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not.

    Returns
    -------
    RDM : array
        The fMRI-Searchlight RDM.
        If sub_result=0, the shape of RDMs is [n_x, n_y, n_z, n_cons, n_cons].
        If sub_result=1, the shape of RDMs is [n_subs, n_x, n_y, n_cons, n_cons]
        n_subs, n_x, n_y, n_z represent the number of subjects & the number of calculation units for searchlight along
        the x, y, z axis.
    """

    # get the number of conditions, subjects and the size of the fMRI-img
    cons, subs, nx, ny, nz = np.shape(fmri_data)

    # the size of the calculation units for searchlight
    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    # strides for calculating along the x, y, z axis
    sx = strides[0]
    sy = strides[1]
    sz = strides[2]

    # calculate the number of the calculation units in the x, y, z directions
    n_x = int((nx - kx) / sx)+1
    n_y = int((ny - ky) / sy)+1
    n_z = int((nz - kz) / sz)+1

    # initialize the data for calculating the RDM
    data = np.full([n_x, n_y, n_z, cons, kx*ky*kz, subs], np.nan)

    # assignment
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                for i in range(cons):

                    index = 0

                    for k1 in range(kx):
                        for k2 in range(ky):
                            for k3 in range(kz):
                                for j in range(subs):
                                    data[x, y, z, i, index, j] = fmri_data[i, j, x+k1, y+k2, z+k3]

                                index = index + 1

    # shape of data: [n_x, n_y, n_z, cons, kx*ky*kz, subs]
    #              ->[subs, n_x, n_y, n_z, cons, kx*ky*kz]
    data = np.transpose(data, (5, 0, 1, 2, 3, 4))

    # flatten the data for different calculating conditions
    data = np.reshape(data, [subs, n_x, n_y, n_z, cons, kx*ky*kz])

    # initialize the RDMs
    subrdms = np.full([subs, n_x, n_y, n_z, cons, cons], np.nan)

    for sub in range(subs):
        for x in range(n_x):
            for y in range(n_y):
                for z in range(n_z):
                    for i in range(cons):
                        for j in range(cons):

                            # no NaN
                            if (np.isnan(data[:, x, y, z, i]).any() == False) and (np.isnan(data[:, x, y, z, j]).any() == False):
                                # calculate the Pearson Coefficient
                                r = pearsonr(data[sub, x, y, z, i], data[sub, x, y, z, j])[0]
                                # calculate the dissimilarity
                                if abs == True:
                                    subrdms[sub, x, y, z, i, j] = limtozero(1 - np.abs(r))
                                else:
                                    subrdms[sub, x, y, z, i, j] = limtozero(1 - r)

    # average the RDMs
    rdms = np.average(subrdms, axis=0)

    if sub_result == 0:
        return rdms
    if sub_result == 1:
        return subrdms


' a function for calculating the RDM based on fMRI data of a ROI '

def fmriRDM_roi(fmri_data, mask_data, sub_result=0, abs=True):

    """
    Calculate the Representational Dissimilarity Matrix - RDM(s) for fMRI data (for ROI)

    Parameters
    ----------
    fmri_data : array
        The fmri data.
        The shape of fmri_data must be [n_cons, n_subs, nx, ny, nz]. n_cons, nx, ny, nz represent the number of
        conditions, the number of subs & the size of fMRI-img, respectively.
    mask_data : array [nx, ny, nz].
        The mask data for region of interest (ROI)
        The size of the fMRI-img. nx, ny, nz represent the number of voxels along the x, y, z axis.
    sub_result: int 0 or 1. Default is 0.
        Return the subject-result or average-result.
        If sub_result=0, return the average result.
        If sub_result=1, return the results of each subject.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not.

    Returns
    -------
    RDM : array
        The fMRI-ROI RDM.
        If sub_result=0, the shape of RDM is [n_cons, n_cons].
        If sub_result=1, the shape of RDM is [n_subs, n_cons, n_cons].
    """

    # get the number of conditions, subjects, the size of the fMRI-img
    ncons, nsubs, nx, ny, nz = fmri_data.shape

    # record the the number of voxels that is not 0 or NaN
    n = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                # not 0 or NaN
                if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False)\
                        and (np.isnan(fmri_data[:, :, :, j, k]).any() == False):
                    n = n + 1

    # initialize the data for calculating the RDM
    data = np.zeros([ncons, nsubs, n], dtype=np.float)

    # assignment
    for p in range(ncons):
        for q in range(nsubs):

            n = 0

            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):

                        # not 0 or NaN
                        if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False)\
                                and (np.isnan(fmri_data[:, :, :, j, k]).any() == False):
                            data[p, q, n] = fmri_data[p, q, i, j, k]
                            n = n + 1

    # initialize the RDMs
    subrdms = np.zeros([nsubs, ncons, ncons], dtype=np.float)

    # shape of data: [ncons, nsubs, n] -> [nsubs, ncons, n]
    data = np.transpose(data, (1, 0, 2))

    # calculate the values in RDM
    for sub in range(nsubs):
        for i in range(ncons):
            for j in range(ncons):

                if (np.isnan(data[:, i]).any() == False) and (np.isnan(data[:, j]).any() == False):
                    # calculate the Pearson Coefficient
                    r = pearsonr(data[sub, i], data[sub, j])[0]
                    # calculate the dissimilarity
                    if abs is True:
                        subrdms[sub, i, j] = limtozero(1 - np.abs(r))
                    else:
                        subrdms[sub, i, j] = limtozero(1 - r)

    # average the RDMs
    rdm = np.average(subrdms, axis=0)

    if sub_result == 0:
        return rdm
    if sub_result == 1:
        return subrdms