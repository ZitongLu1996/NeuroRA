# -*- coding: utf-8 -*-

' a module for calculating the RDM based on multimode neural data '

__author__ = 'Zitong Lu'

import numpy as np
from neurora.stuff import limtozero
import math
from scipy.stats import pearsonr
from neurora.stuff import show_progressbar
from neurora.decoding import tbyt_decoding_kfold

np.seterr(divide='ignore', invalid='ignore')


' a function for calculating the RDM(s) based on behavioral data '

def bhvRDM(bhv_data, sub_opt=1, method="correlation", abs=False):

    """
    Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for behavioral data

    Parameters
    ----------
    bhv_data : array
        The behavioral data.
        The shape of bhv_data must be [n_cons, n_subs, n_trials].
        n_cons, n_subs & n_trials represent the number of conidtions, the number of subjects & the number of trials,
        respectively.
    sub_opt: int 0 or 1. Default is 1.
        Return the results for each subject or after averaging.
        If sub_opt=1, calculate the results of each subject (using the absolute distance).
        If sub_opt=0, calculate the results averaging the trials and taking the subjects as the features.
    method : string 'correlation' or 'euclidean'. Default is 'correlation'.
        The method to calculate the dissimilarities.
        If method='correlation', the dissimilarity is calculated by Pearson Correlation.
        If method='euclidean', the dissimilarity is calculated by Euclidean Distance, the results will be normalized.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not. Only works when method='correlation'.

    Returns
    -------
    RDM(s) : array
        The behavioral RDM.
        If sub_opt=1, return n_subs RDMs. The shape is [n_subs, n_cons, n_cons].
        If sub_opt=0, return only one RDM. The shape is [n_cons, n_cons].

    Notes
    -----
    This function can also be used to calculate the RDM for computational simulation data.
        For example, users can extract the activations for a certain layer i which includes Nn nodes in a deep
        convolutional neural network (DCNN) corresponding to Ni images. Thus, the input could be a [Ni, 1, Nn] matrix M.
        Using "bhvRDM(M, sub_opt=0)", users can obtain the DCNN RDM for layer i.
    """

    if len(np.shape(bhv_data)) != 3:

        print("\nThe shape of input for bhvEEG() function must be [n_cons, n_subs, n_trials].\n")

        return "Invalid input!"

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

        print("\nComputing RDMs")

        # initialize the RDMs
        rdms = np.zeros([subs, cons, cons])

        # calculate the values in RDMs
        for sub in range(subs):
            rdm = np.zeros([cons, cons], dtype=float)
            for i in range(cons):
                for j in range(cons):
                    # calculate the difference
                    if abs == True:
                        rdm[i, j] = np.abs(np.average(bhv_data[i, sub])-np.average(bhv_data[j, sub]))
                    else:
                        rdm[i, j] = np.average(bhv_data[i, sub]) - np.average(bhv_data[j, sub])

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
                            rdm[i, j] = (rdm[i, j] - minvalue) / (maxvalue - minvalue)
            rdms[sub] = rdm

        print("\nRDMs computing finished!")

        return rdms

    # & sub_opt=0

    print("\nComputing RDM")

    # initialize the RDM
    rdm = np.zeros([cons, cons])

    # judge whether numbers of trials of different conditions are same
    if len(set(n_subs)) != 1:
        return None

    # assignment
    # save the data for each subject under each condition, average the trials
    data = np.average(bhv_data, axis=2)

    # calculate the values in RDM
    for i in range(cons):
        for j in range(cons):
            if method == 'correlation':
                # calculate the Pearson Coefficient
                r = pearsonr(data[i], data[j])[0]
                # calculate the dissimilarity
                if abs == True:
                    rdm[i, j] = limtozero(1 - np.abs(r))
                else:
                    rdm[i, j] = limtozero(1 - r)
            elif method == 'euclidean':
                rdm[i, j] = np.linalg.norm(data[i]-data[j])
    if method == 'euclidean':
        max = np.max(rdm)
        min = np.min(rdm)
        rdm = (rdm-min)/(max-min)

    print("\nRDM computing finished!")

    return rdm


' a function for calculating the RDM(s) based on EEG/MEG/fNIRS & other EEG-like data '

def eegRDM(EEG_data, sub_opt=1, chl_opt=0, time_opt=0, time_win=5, time_step=5, method="correlation", abs=False):

    """
    Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) based on EEG-like data

    Parameters
    ----------
    EEG_data : array
        The EEG/MEG/fNIRS data.
        The shape of EEGdata must be [n_cons, n_subs, n_trials, n_chls, n_ts].
        n_cons, n_subs, n_trials, n_chls & n_ts represent the number of conidtions, the number of subjects, the number
        of trials, the number of channels & the number of time-points, respectively.
    sub_opt: int 0 or 1. Default is 1.
        Return the subject-result or average-result.
        If sub_opt=0, return the average result.
        If sub_opt=1, return the results of each subject.
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
    method : string 'correlation' or 'euclidean'. Default is 'correlation'.
        The method to calculate the dissimilarities.
        If method='correlation', the dissimilarity is calculated by Pearson Correlation.
        If method='euclidean', the dissimilarity is calculated by Euclidean Distance, the results will be normalized.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not.

    Returns
    -------
    RDM(s) : array
        The EEG/MEG/fNIR/other EEG-like RDM.
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

    Notes
    -----
    Sometimes, the numbers of trials under different conditions are not same. In NeuroRA, we recommend users to average
    the trials under a same condition firstly in this situation. Thus, the shape of input (EEG_data) should be
    [n_cons, n_subs, 1, n_chls, n_ts].
    """

    if len(np.shape(EEG_data)) != 5:

        print("The shape of input for eegRDM() function must be [n_cons, n_subs, n_trials, n_chls, n_ts].\n")

        return "Invalid input!"

    # get the number of conditions, subjects, trials, channels and time points
    cons, subs, trials, chls, ts = np.shape(EEG_data)

    if time_opt == 1:

        print("\nComputing RDMs")

        # the time-points for calculating RDM
        ts = int((ts - time_win) / time_step) + 1

        # initialize the data for calculating the RDM
        data = np.zeros([subs, chls, ts, cons, time_win])

        # assignment
        for i in range(subs):
            for j in range(chls):
                for k in range(ts):
                    for l in range(cons):
                        for m in range(time_win):
                            # average the trials
                            data[i, j, k, l, m] = np.average(EEG_data[l, i, :, j, k * time_step + m])

        if chl_opt == 1:

            total = subs*chls*ts

            # initialize the RDMs
            rdms = np.zeros([subs, chls, ts, cons, cons])

            # calculate the values in RDMs
            for i in range(subs):
                for j in range(chls):
                    for k in range(ts):

                        # show the progressbar
                        percent = (i * chls * ts + j * ts + k + 1) / total * 100
                        show_progressbar("Calculating", percent)

                        for l in range(cons):
                            for m in range(cons):
                                if method is 'correlation':
                                    # calculate the Pearson Coefficient
                                    r = pearsonr(data[i, j, k, l], data[i, j, k, m])[0]
                                    # calculate the dissimilarity
                                    if abs == True:
                                        rdms[i, j, k, l, m] = limtozero(1 - np.abs(r))
                                    else:
                                        rdms[i, j, k, l, m] = limtozero(1 - r)
                                elif method == 'euclidean':
                                    rdms[i, j, k, l, m] = np.linalg.norm(data[i, j, k, l] - data[i, j, k, m])
                                """elif method == 'mahalanobis':
                                    X = np.transpose(np.vstack((data[i, j, k, l], data[i, j, k, m])), (1, 0))
                                    X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
                                    rdms[i, j, k, l, m] = np.linalg.norm(X[:, 0] - X[:, 1])"""
                        if method == 'euclidean':
                            max = np.max(rdms[i, j, k])
                            min = np.min(rdms[i, j, k])
                            rdms[i, j, k] = (rdms[i, j, k] - min) / (max - min)

            # time_opt=1 & chl_opt=1 & sub_opt=1
            if sub_opt == 1:

                print("\nRDMs computing finished!")

                return rdms

            # time_opt=1 & chl_opt=1 & sub_opt=0
            if sub_opt == 0:

                rdms = np.average(rdms, axis=0)

                print("\nRDMs computing finished!")

                return rdms

        # if chl_opt = 0

        data = np.transpose(data, (0, 2, 3, 4, 1))
        data = np.reshape(data, [subs, ts, cons, time_win*chls])

        rdms = np.zeros([subs, ts, cons, cons])

        total = subs * ts

        # calculate the values in RDMs
        for i in range(subs):
            for k in range(ts):

                # show the progressbar
                percent = (i * ts + k + 1) / total * 100
                show_progressbar("Calculating", percent)

                for l in range(cons):
                    for m in range(cons):
                        if method == 'correlation':
                            # calculate the Pearson Coefficient
                            r = pearsonr(data[i, k, l], data[i, k, m])[0]
                            # calculate the dissimilarity
                            if abs is True:
                                rdms[i, k, l, m] = limtozero(1 - np.abs(r))
                            else:
                                rdms[i, k, l, m] = limtozero(1 - r)
                        elif method == 'euclidean':
                            rdms[i, k, l, m] = np.linalg.norm(data[i, k, l] - data[i, k, m])
                if method == 'euclidean':
                    max = np.max(rdms[i, k])
                    min = np.min(rdms[i, k])
                    rdms[i, k] = (rdms[i, k] - min) / (max - min)

        # time_opt=1 & chl_opt=0 & sub_opt=1
        if sub_opt == 1:

            print("\nRDMs computing finished!")

            return rdms

        # time_opt=1 & chl_opt=0 & sub_opt=0
        if sub_opt == 0:

            rdms = np.average(rdms, axis=0)

            print("\nRDM computing finished!")

            return rdms


    # if time_opt = 0

    if chl_opt == 1:

        print("\nComputing RDMs")

        # average the trials
        data = np.average(EEG_data, axis=2)

        print(data.shape)

        # initialize the RDMs
        rdms = np.zeros([subs, chls, cons, cons])

        total = subs * chls

        # calculate the values in RDMs
        for i in range(subs):
            for j in range(chls):

                # show the progressbar
                percent = (i * chls + j + 1) / total * 100
                show_progressbar("Calculating", percent)

                for k in range(cons):
                    for l in range(cons):
                        if method == 'correlation':
                            # calculate the Pearson Coefficient
                            r = pearsonr(data[k, i, j], data[l, i, j])[0]
                            # calculate the dissimilarity
                            if abs == True:
                                rdms[i, j, k, l] = limtozero(1 - np.abs(r))
                            else:
                                rdms[i, j, k, l] = limtozero(1 - r)
                        elif method == 'euclidean':
                            rdms[i, j, k, l] = np.linalg.norm(data[k, i, j] - data[l, i, j])
                if method == 'euclidean':
                    max = np.max(rdms[i, j])
                    min = np.min(rdms[i, j])
                    rdms[i, j] = (rdms[i, j] - min) / (max - min)

        # time_opt=0 & chl_opt=1 & sub_opt=1
        if sub_opt == 1:

            print("\nRDM computing finished!")

            return rdms

        # time_opt=0 & chl_opt=1 & sub_opt=0
        if sub_opt == 0:

            rdms = np.average(rdms, axis=0)

            print("\nRDM computing finished!")

            return rdms

    # if chl_opt = 0

    if sub_opt == 1:

        print("\nComputing RDMs")

    else:

        print("\nComputing RDM")

    # average the trials
    data = np.average(EEG_data, axis=2)

    # flatten the data for different calculating conditions
    data = np.reshape(data, [cons, subs, chls * ts])

    # initialize the RDMs
    rdms = np.zeros([subs, cons, cons])

    # calculate the values in RDMs
    for i in range(subs):
        for j in range(cons):
            for k in range(cons):
                if method == 'correlation':
                    # calculate the Pearson Coefficient
                    r = pearsonr(data[j, i], data[k, i])[0]
                    # calculate the dissimilarity
                    if abs == True:
                        rdms[i, j, k] = limtozero(1 - np.abs(r))
                    else:
                        rdms[i, j, k] = limtozero(1 - r)
                elif method == 'euclidean':
                    rdms[i, j, k] = np.linalg.norm(data[j, i] - data[k, i])
                """elif method == 'mahalanobis':
                    X = np.transpose(np.vstack((data[j, i], data[k, i])), (1, 0))
                    X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
                    rdms[i, j, k] = np.linalg.norm(X[:, 0] - X[:, 1])"""
        if method == 'euclidean':
            max = np.max(rdms[i])
            min = np.min(rdms[i])
            rdms[i] = (rdms[i] - min) / (max - min)

    if sub_opt == 1:

        print("\nRDMs computing finished!")

        return rdms

    if sub_opt == 0:

        rdms = np.average(rdms, axis=0)

        print("\nRDM computing finished!")

        return rdms


' a function for calculating the RDM(s) using classification-based neural decoding based on EEG/MEG/fNIRS & other EEG-like data '

def eegRDM_bydecoding(EEG_data, sub_opt=1, time_win=5, time_step=5, navg=5, time_opt="average", nfolds=5, nrepeats=2,
                      normalization=False):

    """
    Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) using classification-based neural decoding
    based on EEG-like data

    Parameters
    ----------
    EEG_data : array
        The EEG/MEG/fNIRS data.
        The shape of EEGdata must be [n_cons, n_subs, n_trials, n_chls, n_ts].
        n_cons, n_subs, n_trials, n_chls & n_ts represent the number of conidtions, the number of subjects, the number
        of trials, the number of channels & the number of time-points, respectively.
    sub_opt: int 0 or 1. Default is 1.
        Return the subject-result or average-result.
        If sub_opt=0, return the average result.
        If sub_opt=1, return the results of each subject.
    time_win : int. Default is 5.
        Set a time-window for calculating the RDM for different time-points.
        Only when time_opt=1, time_win works.
        If time_win=5, that means each calculation process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.
        Only when time_opt=1, time_step works.
    navg : int. Default is 5.
        The number of trials used to average.
    time_opt : string "average" or "features". Default is "average".
        Average the time-points or regard the time points as features for classification
        If time_opt="average", the time-points in a certain time-window will be averaged.
        If time_opt="features", the time-points in a certain time-window will be used as features for classification.
    nfolds : int. Default is 5.
        The number of folds.
        k should be at least 2.
    nrepeats : int. Default is 2.
        The times for iteration.
    normalization : boolean True or False. Default is False.
        Normalize the data or not.

    Returns
    -------
    RDM(s) : array
        The EEG/MEG/fNIR/other EEG-like RDM.
        If sub_opt=0, return int((n_ts-time_win)/time_step)+1 RDMs.
            The shape is [int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
        If sub_opt=1, return n_subs*int((n_ts-time_win)/time_step)+1 RDM.
            The shape is [n_subs, int((n_ts-time_win)/time_step)+1, n_cons, n_cons].

    Notes
    -----
    Sometimes, the numbers of trials under different conditions are not same. In NeuroRA, we recommend users to sample
    randomly from the trials under each conditions to keep the numbers of trials under different conditions same, and
    you can iterate multiple times.
    """

    if len(np.shape(EEG_data)) != 5:

        print("The shape of input for eegRDM() function must be [n_cons, n_subs, n_trials, n_chls, n_ts].\n")

        return "Invalid input!"

    # get the number of conditions, subjects, trials, channels and time points
    cons, subs, trials, chls, ts = np.shape(EEG_data)

    ts = int((ts - time_win) / time_step) + 1

    rdms = np.zeros([subs, ts, cons, cons])

    for con1 in range(cons):
        for con2 in range(cons):

            if con1 > con2:

                data = np.concatenate((EEG_data[con1], EEG_data[con2]), axis=1)
                labels = np.zeros([subs, 2*trials])
                labels[:, trials:] = 1
                rdms[:, :, con1, con2] = tbyt_decoding_kfold(data, labels, n=2, navg=navg, time_opt=time_opt,
                                                             time_win=time_win, time_step=time_step, nfolds=nfolds,
                                                             nrepeats=nrepeats, normalization=normalization,
                                                             pca=False, smooth=True)
                rdms[:, :, con2, con1] = rdms[:, :, con1, con2]

    if sub_opt == 0:

        return np.average(rdms, axis=0)

    else:

        return rdms


' a function for calculating the RDMs based on fMRI data (searchlight) '

def fmriRDM(fmri_data, ksize=[3, 3, 3], strides=[1, 1, 1], sub_opt=1, method="correlation", abs=False):

    """
    Calculate the Representational Dissimilarity Matrices (RDMs) based on fMRI data (searchlight)

    Parameters
    ----------
    fmri_data : array
        The fmri data.
        The shape of fmri_data must be [n_cons, n_subs, nx, ny, nz]. n_cons, nx, ny, nz represent the number of
        conditions, the number of subs & the size of fMRI-img, respectively.
    ksize : array or list [kx, ky, kz]. Default is [3, 3, 3].
        The size of the calculation unit for searchlight.
        kx, ky, kz represent the number of voxels along the x, y, z axis.
        kx, ky, kz should be odd.
    strides : array or list [sx, sy, sz]. Default is [1, 1, 1].
        The strides for calculating along the x, y, z axis.
    sub_opt: int 0 or 1. Default is 1.
        Return the subject-result or average-result.
        If sub_opt=0, return the average result.
        If sub_opt=1, return the results of each subject.
    method : string 'correlation' or 'euclidean'. Default is 'correlation'.
        The method to calculate the dissimilarities.
        If method='correlation', the dissimilarity is calculated by Pearson Correlation.
        If method='euclidean', the dissimilarity is calculated by Euclidean Distance, the results will be normalized.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not.

    Returns
    -------
    RDM : array
        The fMRI-Searchlight RDM.
        If sub_opt=0, the shape of RDMs is [n_x, n_y, n_z, n_cons, n_cons].
        If sub_opt=1, the shape of RDMs is [n_subs, n_x, n_y, n_cons, n_cons]
        n_subs, n_x, n_y, n_z represent the number of subjects & the number of calculation units for searchlight along
        the x, y, z axis.
    """

    if len(np.shape(fmri_data)) != 5:

        print("\nThe shape of input for fmriRDM() function must be [n_cons, n_subs, nx, ny, nz].\n")

        return "Invalid input!"

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

    print("\nComputing RDMs")

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
                                    data[x, y, z, i, index, j] = fmri_data[i, j, x*sx+k1, y*sy+k2, z*sz+k3]

                                index = index + 1

    # shape of data: [n_x, n_y, n_z, cons, kx*ky*kz, subs]
    #              ->[subs, n_x, n_y, n_z, cons, kx*ky*kz]
    data = np.transpose(data, (5, 0, 1, 2, 3, 4))

    # flatten the data for different calculating conditions
    data = np.reshape(data, [subs, n_x, n_y, n_z, cons, kx*ky*kz])

    # initialize the RDMs
    subrdms = np.full([subs, n_x, n_y, n_z, cons, cons], np.nan)

    total = subs * n_x * n_y * n_z

    for sub in range(subs):
        for x in range(n_x):
            for y in range(n_y):
                for z in range(n_z):

                    # show the progressbar
                    percent = (sub * n_x * n_y * n_z + x * n_y * n_z + y * n_z + z + 1) / total * 100
                    show_progressbar("Calculating", percent)

                    for i in range(cons):
                        for j in range(cons):

                            # no NaN
                            if (np.isnan(data[:, x, y, z, i]).any() == False) and \
                                    (np.isnan(data[:, x, y, z, j]).any() == False):
                                if method == 'correlation':
                                    # calculate the Pearson Coefficient
                                    r = pearsonr(data[sub, x, y, z, i], data[sub, x, y, z, j])[0]
                                    # calculate the dissimilarity
                                    if abs == True:
                                        subrdms[sub, x, y, z, i, j] = limtozero(1 - np.abs(r))
                                    else:
                                        subrdms[sub, x, y, z, i, j] = limtozero(1 - r)
                                elif method == 'euclidean':
                                    subrdms[sub, x, y, z, i, j] = np.linalg.norm(data[sub, x, y, z, i] -
                                                                                 data[sub, x, y, z, j])
                                """elif method == 'mahalanobis':
                                    X = np.transpose(np.vstack((data[sub, x, y, z, i], data[sub, x, y, z, j])), (1, 0))
                                    X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
                                    subrdms[sub, x, y, z, i, j] = np.linalg.norm(X[:, 0] - X[:, 1])"""
                    if method == 'euclidean':
                        max = np.max(subrdms[sub, x, y, z])
                        min = np.min(subrdms[sub, x, y, z])
                        subrdms[sub, x, y, z] = (subrdms[sub, x, y, z] - min) / (max - min)

    # average the RDMs
    rdms = np.average(subrdms, axis=0)

    print("\nRDMs computing finished!")

    if sub_opt == 0:

        return rdms

    if sub_opt == 1:

        return subrdms


' a function for calculating the RDM based on fMRI data of an ROI '

def fmriRDM_roi(fmri_data, mask_data, sub_opt=1, method="correlation", abs=False):

    """
    Calculate the Representational Dissimilarity Matrix - RDM(s) based on fMRI data (for ROI)

    Parameters
    ----------
    fmri_data : array
        The fmri data.
        The shape of fmri_data must be [n_cons, n_subs, nx, ny, nz]. n_cons, nx, ny, nz represent the number of
        conditions, the number of subs & the size of fMRI-img, respectively.
    mask_data : array [nx, ny, nz].
        The mask data for region of interest (ROI)
        The size of the fMRI-img. nx, ny, nz represent the number of voxels along the x, y, z axis.
    sub_opt: int 0 or 1. Default is 1.
        Return the subject-result or average-result.
        If sub_opt=0, return the average result.
        If sub_opt=1, return the results of each subject.
    method : string 'correlation' or 'euclidean'. Default is 'correlation'.
        The method to calculate the dissimilarities.
        If method='correlation', the dissimilarity is calculated by Pearson Correlation.
        If method='euclidean', the dissimilarity is calculated by Euclidean Distance, the results will be normalized.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not.

    Returns
    -------
    RDM : array
        The fMRI-ROI RDM.
        If sub_opt=0, the shape of RDM is [n_cons, n_cons].
        If sub_opt=1, the shape of RDM is [n_subs, n_cons, n_cons].

    Notes
    -----
    The sizes (nx, ny, nz) of fmri_data and mask_data should be same.
    """

    if len(np.shape(fmri_data)) != 5 or len(np.shape(mask_data)) != 3:

        print("\nThe shape of inputs (fmri_data & mask_data) for fmriRDM_roi() function should be [n_cons, "
              "n_subs, nx, ny, nz] & [nx, ny, nz], respectively.\n")

        return "Invalid input!"

    # get the number of conditions, subjects, the size of the fMRI-img
    ncons, nsubs, nx, ny, nz = fmri_data.shape

    # record the the number of voxels that is not 0 or NaN
    n = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                # not 0 or NaN
                if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False)\
                        and (np.isnan(fmri_data[:, :, i, j, k]).any() == False):
                    n = n + 1

    # initialize the data for calculating the RDM
    data = np.zeros([ncons, nsubs, n])

    print("\nComputing RDMs")

    # assignment
    for p in range(ncons):
        for q in range(nsubs):

            n = 0

            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):

                        # not 0 or NaN
                        if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False)\
                                and (np.isnan(fmri_data[:, :, i, j, k]).any() == False):
                            data[p, q, n] = fmri_data[p, q, i, j, k]
                            n = n + 1

    # initialize the RDMs
    subrdms = np.zeros([nsubs, ncons, ncons])

    # shape of data: [ncons, nsubs, n] -> [nsubs, ncons, n]
    data = np.transpose(data, (1, 0, 2))

    # calculate the values in RDM
    for sub in range(nsubs):
        for i in range(ncons):
            for j in range(ncons):

                if (np.isnan(data[:, i]).any() == False) and (np.isnan(data[:, j]).any() == False):
                    if method == 'correlation':
                        # calculate the Pearson Coefficient
                        r = pearsonr(data[sub, i], data[sub, j])[0]
                        # calculate the dissimilarity
                        if abs == True:
                            subrdms[sub, i, j] = limtozero(1 - np.abs(r))
                        else:
                            subrdms[sub, i, j] = limtozero(1 - r)
                    elif method == 'euclidean':
                        subrdms[sub, i, j] = np.linalg.norm(data[sub, i] - data[sub, j])
                    """elif method == 'mahalanobis':
                        X = np.transpose(np.vstack((data[sub, i], data[sub, j])), (1, 0))
                        X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
                        subrdms[sub, i, j] = np.linalg.norm(X[:, 0] - X[:, 1])"""
        if method == 'euclidean':
            max = np.max(subrdms[sub])
            min = np.min(subrdms[sub])
            subrdms[sub] = (subrdms[sub] - min) / (max - min)

    # average the RDMs
    rdm = np.average(subrdms, axis=0)

    if sub_opt == 0:

        print("\nRDM computing finished!")

        return rdm

    if sub_opt == 1:

        print("\nRDMs computing finished!")

        return subrdms