# -*- coding: utf-8 -*-

' a module for calculating the Similarity/Correlation Coefficient between two different modes data '

__author__ = 'Zitong Lu'

import numpy as np
from neurora.rdm_cal import bhvRDM
from neurora.rdm_cal import eegRDM
from neurora.rdm_cal import fmriRDM
from neurora.rdm_corr import rdm_correlation_spearman
from neurora.rdm_corr import rdm_correlation_pearson
from neurora.rdm_corr import rdm_correlation_kendall
from neurora.rdm_corr import rdm_similarity
from neurora.rdm_corr import rdm_distance

np.seterr(divide='ignore', invalid='ignore')


' a function for calculating the Similarity/Correlation Coefficient between behavioral data and EEG/MEG/fNIRS data'

def bhvANDeeg_corr(bhv_data, eeg_data, sub_opt=0, chl_opt=0, time_opt=0, time_win=5, time_step=5, method="spearman", fisherz=False, rescale=False, permutation=False, iter=5000):

    """
    Calculate the Similarities between behavioral data and EEG/MEG/fNIRS data

    Parameters
    ----------
    bhv_data : array
        The behavioral data.
        The shape of bhv_data must be [n_cons, n_subs, n_trials].
        n_cons, n_subs & n_trials represent the number of conidtions, the number of subjects & the number of trials,
        respectively.
    eeg_data : array
        The EEG/MEG/fNIRS data.
        The shape of EEGdata must be [n_cons, n_subs, n_trials, n_chls, n_ts].
        n_cons, n_subs, n_trials, n_chls & n_ts represent the number of conidtions, the number of subjects, the number
        of trials, the number of channels & the number of time-points, respectively.
    sub_opt : int 0 or 1. Default is 0.
        Calculate the RDM & similarities for each subject or not.
        If sub_opt=0, calculating based on all data.
        If sub_opt=1, calculating based on each subject's data, respectively.
    chl_opt : int 0 or 1. Default is 0.
        Calculate the RDM & similarities for each channel or not.
        If chl_opt=0, calculating based on all channels' data.
        If chl_opt=1, calculating based on each channel's data respectively.
    time_opt : int 0 or 1. Default is 0.
        Calculate the RDM & similarities for each time-point or not
        If time_opt=0, calculating based on whole time-points' data.
        If time_opt=1, calculating based on each time-points respectively.
    time_win : int. Default is 5.
        Set a time-window for calculating the RDM & similarities for different time-points.
        Only when time_opt=1, time_win works.
        If time_win=5, that means each calculation process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.
        Only when time_opt=1, time_step works.
    method : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or 'distance'. Default is 'spearman'.
        The method to calculate the similarities.
        If method='spearman', calculate the Spearman Correlations. If method='pearson', calculate the Pearson
        Correlations. If methd='kendall', calculate the Kendall tau Correlations. If method='similarity', calculate the
        Cosine Similarities. If method='distance', calculate the Euclidean Distances.
    fisherz : bool True or False. Default is False.
        Do the Fisher-Z transform of the RDMs or not.
    rescale : bool True or False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagonal.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    corrs : array
        The similarities between behavioral data and EEG/MEG/fNIRS data.
        If sub_opt=0 & chl_opt=0 & time_opt=0, return one corr result.
            The shape of corrs is [2], a r-value and a p-value. If method='similarity' or method='distance', the
            p-value is 0.
        If sub_opt=0 & chl_opt=0 & time_opt=1, return int((n_ts-time_win)/time_step)+1 corrs result.
            The shape of corrs is int((n_ts-time_win)/time_step)+1. 2 represents a r-value and a p-value. If
            method='similarity' or method='distance', the p-values are all 0.
        If sub_opt=0 & chl_opt=1 & time_opt=0, return n_chls corrs result.
            The shape of corrs is [n_chls, 2]. 2 represents a r-value and a p-value. If method='similarity' or
            method='distance', the p-values are all 0.
        If sub_opt=0 & chl_opt=1 & time_opt=1, return n_chls*(int((n_ts-time_win)/time_step)+1) corrs result.
            The shape of corrs is [n_chls, int((n_ts-time_win)/time_step)+1, 2]. 2 represents a r-value and a p-value.
            If method='similarity' or method='distance', the p-values are all 0.
        If sub_opt=1 & chl_opt=0 & time_opt=0, return n_subs corr result.
            The shape of corrs is [n_subs, 2], a r-value and a p-value. If method='similarity' or method='distance',
            the p-values are all 0.
        If sub_opt=1 & chl_opt=0 & time_opt=1, return n_subs*(int((n_ts-time_win)/time_step)+1) corrs result.
            The shape of corrs is [n_subs, int((n_ts-time_win)/time_step)+1, 2]. 2 represents a r-value and a p-value.
            If method='similarity' or method='distance', the p-values are all 0.
        If sub_opt=1 & chl_opt=1 & time_opt=0, return n_subs*n_chls corrs result.
            The shape of corrs is [n_subs, n_chls, 2]. 2 represents a r-value and a p-value. If method='similarity' or
            method='distance', the p-values are all 0.
        If sub_opt=1 & chl_opt=1 & time_opt=1, return n_subs*n_chls*(int((n_ts-time_win)/time_step)+1) corrs result.
            The shape of corrs is [n_subs, n_chls, int((n_ts-time_win)/time_step)+1, 2]. 2 represents a r-value and a
            p-value. If method='similarity' or method='distance', the p-values are all 0.
    """

    # get the number of subjects
    subs = np.shape(bhv_data)[1]
    # get the number of channels
    chls = np.shape(eeg_data)[3]
    # get the number of time-points for calculating
    ts = np.shape(eeg_data)[4]
    ts = int((ts-time_win)/time_step)+1

    if sub_opt == 1:

        # calculate the bhv_rdm
        bhv_rdms = bhvRDM(bhv_data, sub_opt=sub_opt)

        if chl_opt == 0:

            if time_opt == 0:

                # sub_opt=1 & chl_opt=0 & time_opt=0

                # calculate the eeg_rdms
                eeg_rdms = eegRDM(eeg_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt)

                # initialize the corrs
                corrs = np.zeros([subs, 2], dtype=np.float64)

                # calculate the corrs
                for i in range(subs):

                    if method == "spearman":
                        corrs[i] = rdm_correlation_spearman(bhv_rdms[i], eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "pearson":
                        corrs[i] = rdm_correlation_pearson(bhv_rdms[i], eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "kendall":
                        corrs[i] = rdm_correlation_kendall(bhv_rdms[i], eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "similarity":
                        corrs[i, 0] = rdm_similarity(bhv_rdms[i], eeg_rdms[i], rescale=rescale)
                    elif method == "distance":
                        corrs[i, 0] = rdm_distance(bhv_rdms[i], eeg_rdms[i], rescale=rescale)

                return corrs

            # sub_opt=1 & chl_opt=0 & time_opt=1

            # calculate the eeg_rdms
            eeg_rdms = eegRDM(eeg_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt, time_win=time_win, time_step=time_step)

            # initialize the corrs
            corrs = np.zeros([subs, ts, 2], dtype=np.float64)

            # calculate the corrs
            for i in range(subs):
                for j in range(ts):

                    if method == "spearman":
                        corrs[i, j] = rdm_correlation_spearman(bhv_rdms[i], eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "pearson":
                        corrs[i, j] = rdm_correlation_pearson(bhv_rdms[i], eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "kendall":
                        corrs[i, j] = rdm_correlation_kendall(bhv_rdms[i], eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "similarity":
                        corrs[i, j, 0] = rdm_similarity(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)
                    elif method == "distance":
                        corrs[i, j, 0] = rdm_distance(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

            return corrs

        if time_opt == 1:

            #  sub_opt=1 & chl_opt=1 & time_opt=1

            # calculate the eeg_rdms
            eeg_rdms = eegRDM(eeg_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt, time_win=time_win, time_step=time_step)

            # initialize the corrs
            corrs = np.zeros([subs, chls, ts, 2], dtype=np.float64)

            # calculate the corrs
            for i in range(subs):
                for j in range(chls):
                    for k in range(ts):

                        if method == "spearman":
                            corrs[i, j, k] = rdm_correlation_spearman(bhv_rdms[i], eeg_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                        elif method == "pearson":
                            corrs[i, j, k] = rdm_correlation_pearson(bhv_rdms[i], eeg_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                        elif method == "kendall":
                            corrs[i, j, k] = rdm_correlation_kendall(bhv_rdms[i], eeg_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                        elif method == "similarity":
                            corrs[i, j, k, 0] = rdm_similarity(bhv_rdms[i], eeg_rdms[i, j, k], rescale=rescale)
                        elif method == "distance":
                            corrs[i, j, k, 0] = rdm_distance(bhv_rdms[i], eeg_rdms[i, j, k], rescale=rescale)

            return corrs


        # sub_opt=1 & chl_opt=1 & time_opt=0

        eeg_rdms = eegRDM(eeg_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt)

        corrs = np.zeros([subs, chls], dtype=np.float64)

        for i in range(subs):
            for j in range(chls):

                if method == "spearman":
                    corrs[i, j] = rdm_correlation_spearman(bhv_rdms[i], eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "pearson":
                    corrs[i, j] = rdm_correlation_pearson(bhv_rdms[i], eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "kendall":
                    corrs[i, j] = rdm_correlation_kendall(bhv_rdms[i], eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "similarity":
                    corrs[i, j, 0] = rdm_similarity(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)
                elif method == "distance":
                    corrs[i, j, 0] = rdm_distance(bhv_rdms[i], eeg_rdms[i, j], rescale=rescale)

        return corrs

    # if sub_opt=0

    bhv_rdm = bhvRDM(bhv_data, sub_opt=sub_opt)

    if chl_opt == 1:

        if time_opt == 1:

            # sub_opt = 0 & chl_opt = 1 & time_opt = 1

            # calculate the eeg_rdms
            eeg_rdms = eegRDM(eeg_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt, time_win=time_win, time_step=time_step)

            # initialize the corrs
            corrs = np.zeros([chls, ts, 2], dtype=np.float64)

            # calculate the corrs
            for i in range(chls):
                for j in range(ts):

                    if method == "spearman":
                        corrs[i, j] = rdm_correlation_spearman(bhv_rdm, eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "pearson":
                        corrs[i, j] = rdm_correlation_pearson(bhv_rdm, eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "kendall":
                        corrs[i, j] = rdm_correlation_kendall(bhv_rdm, eeg_rdms[i, j], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "similarity":
                        corrs[i, j, 0] = rdm_similarity(bhv_rdm, eeg_rdms[i, j], rescale=rescale)
                    elif method == "distance":
                        corrs[i, j, 0] = rdm_distance(bhv_rdm, eeg_rdms[i, j], rescale=rescale)

            return corrs

        # sub_opt=0 & chl_opt=1 & time_opt=0

        # calculate the eeg_rdms
        eeg_rdms = eegRDM(eeg_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt)

        # initialize the corrs
        corrs = np.zeros([chls, 2], dtype=np.float64)

        # calculate the corrs
        for i in range(chls):

            if method == "spearman":
                corrs[i] = rdm_correlation_spearman(bhv_rdm, eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
            elif method == "pearson":
                corrs[i] = rdm_correlation_pearson(bhv_rdm, eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
            elif method == "kendall":
                corrs[i] = rdm_correlation_kendall(bhv_rdm, eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
            elif method == "similarity":
                corrs[i, 0] = rdm_similarity(bhv_rdm, eeg_rdms[i], rescale=rescale)
            elif method == "distance":
                corrs[i, 0] = rdm_distance(bhv_rdm, eeg_rdms[i], rescale=rescale)

        return corrs

    # if chl_opt=0

    if time_opt == 1:

        # sub_opt=0 & chl_opt=0 & time_opt=1

        # calculate the eeg_rdms
        eeg_rdms = eegRDM(eeg_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt, time_win=time_win, time_step=time_step)

        # initialize the corrs
        corrs = np.zeros([ts, 2], dtype=np.float64)

        # calculate the corrs
        for i in range(ts):

            if method == "spearman":
                corrs[i] = rdm_correlation_spearman(bhv_rdm, eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
            elif method == "pearson":
                corrs[i] = rdm_correlation_pearson(bhv_rdm, eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
            elif method == "kendall":
                corrs[i] = rdm_correlation_kendall(bhv_rdm, eeg_rdms[i], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
            elif method == "similarity":
                corrs[i, 0] = rdm_similarity(bhv_rdm, eeg_rdms[i], rescale=rescale)
            elif method == "distance":
                corrs[i, 0] = rdm_distance(bhv_rdm, eeg_rdms[i], rescale=rescale)

        return corrs

    # sub_opt=0 & chl_opt=0 & time_opt=0

    # calculate the eeg_rdms
    eeg_rdm = eegRDM(eeg_data, sub_opt=sub_opt, chl_opt=chl_opt, time_opt=time_opt)

    # initialize the corrs
    corr = np.zeros([2], dtype=np.float64)

    # calculate the corrs
    if method == "spearson":
        corr = rdm_correlation_spearman(bhv_rdm, eeg_rdm, fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
    elif method == "pearson":
        corr = rdm_correlation_pearson(bhv_rdm, eeg_rdm, fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
    elif method == "kendall":
        corr = rdm_correlation_kendall(bhv_rdm, eeg_rdm, fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
    elif method == "similarity":
        corr[0] = rdm_similarity(bhv_rdm, eeg_rdm, rescale=rescale)
    elif method == "distance":
        corr[0] = rdm_distance(bhv_rdm, eeg_rdm, rescale=rescale)

    return corr


' a function for calculating the Similarity/Correlation Cosfficient between behavioral data and fMRI data (searchlight) '

def bhvANDfmri_corr(bhv_data, fmri_data, ksize=[3, 3, 3], strides=[1, 1, 1], sub_result=1, method="spearman", fisherz=False, rescale=False, permutation=False, iter=5000):

    """
    Calculate the Similarities between behavioral data and fMRI data for searchlight

    Parameters
    ----------
    bhv_data : array
        The behavioral data.
        The shape of bhv_data must be [n_cons, n_subs, n_trials].
        n_cons, n_subs & n_trials represent the number of conidtions, the number of subjects & the number of trials,
        respectively.
    fmri_data : array
        The fmri data.
        The shape of fmri_data must be [n_cons, n_subs, nx, ny, nz]. n_cons, nx, ny, nz represent the number of
        conditions, the number of subs & the size of fMRI-img, respectively.
    ksize : array or list [kx, ky, kz]. Default is [3, 3, 3].
        The size of the calculation unit for searchlight.
        kx, ky, kz represent the number of voxels along the x, y, z axis.
    strides : array or list [sx, sy, sz]. Default is [1, 1, 1].
        The strides for calculating along the x, y, z axis.
    sub_result: int 0 or 1. Default is 1.
        Return the subject-result or average-result.
        If sub_result=0, return the average result.
        If sub_result=1, return the results of each subject.
    method : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or 'distance'. Default is 'spearman'.
        The method to calculate the similarities.
        If method='spearman', calculate the Spearman Correlations. If method='pearson', calculate the Pearson
        Correlations. If methd='kendall', calculate the Kendall tau Correlations. If method='similarity', calculate the
        Cosine Similarities. If method='distance', calculate the Euclidean Distances.
    fisherz : bool True or False. Default is False.
        Do the Fisher-Z transform of the RDMs or not.
    rescale : bool True or False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the
        values on the diagonal.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    corrs : array
        The similarities between behavioral data and fMRI data for searchlight.
        If sub_result=0, the shape of corrs is [n_x, n_y, n_z, 2].
        If sub_result=1, the shape of corrs is [n_subs, n_x, n_y, n_z, 2].
        n_x, n_y, n_z represent the number of calculation units for searchlight along the x, y, z axis and 2 represents
        a r-value and a p-value.
    """

    # calculate the bhv_rdm
    bhv_rdm = bhvRDM(bhv_data, sub_opt=0)

    print("****************")
    print("get behavior RDM")
    print(bhv_rdm)

    # calculate the fmri_rdms for searchlight
    fmri_rdms = fmriRDM(fmri_data, ksize=ksize, strides=strides, sub_result=sub_result)

    print("****************")
    print("get fMRI RDM")
    print(np.shape(fmri_rdms))

    # get the number of subjects
    subs = np.shape(fmri_data)[1]

    # get the size of the fMRI-img
    nx = np.shape(fmri_data)[2]
    ny = np.shape(fmri_data)[3]
    nz = np.shape(fmri_data)[4]

    # the size of the calculation units for searchlight
    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    # strides for calculating along the x, y, z axis
    sx = strides[0]
    sy = strides[1]
    sz = strides[2]

    # calculate the number of the calculation units in the x, y, z directions
    n_x = int((nx - kx) / sx) + 1
    n_y = int((ny - ky) / sy) + 1
    n_z = int((nz - kz) / sz) + 1

    # sub_result=0
    if sub_result == 0:

        # initialize the corrs
        corrs = np.full([n_x, n_y, n_z, 2], np.nan)

        # calculate the corrs
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):

                    if method == "spearman":
                        corrs[i, j, k] = rdm_correlation_spearman(bhv_rdm, fmri_rdms[i, j, k], rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "pearson":
                        corrs[i, j, k] = rdm_correlation_pearson(bhv_rdm, fmri_rdms[i, j, k], rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "kendall":
                        corrs[i, j, k] = rdm_correlation_kendall(bhv_rdm, fmri_rdms[i, j, k], rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "similarity":
                        corrs[i, j, k, 0] = rdm_similarity(bhv_rdm, fmri_rdms[i, j, k], rescale=rescale)
                    elif method == "distance":
                        corrs[i, j, k, 0] = rdm_distance(bhv_rdm, fmri_rdms[i, j, k], rescale=rescale)

                    print(corrs[i, j, k])

        return corrs

    # sub_result=1

    # initialize the corrs
    corrs = np.full([subs, n_x, n_y, n_z, 2], np.nan)

    # calculate the corrs
    for sub in range(subs):
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):

                    if method == "spearman":
                        corrs[sub, i, j, k] = rdm_correlation_spearman(bhv_rdm, fmri_rdms[sub, i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "pearson":
                        corrs[sub, i, j, k] = rdm_correlation_pearson(bhv_rdm, fmri_rdms[sub, i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "kendall":
                        corrs[sub, i, j, k] = rdm_correlation_kendall(bhv_rdm, fmri_rdms[sub, i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                    elif method == "similarity":
                        corrs[sub, i, j, k, 0] = rdm_similarity(bhv_rdm, fmri_rdms[sub, i, j, k], rescale=rescale)
                    elif method == "distance":
                        corrs[sub, i, j, k, 0] = rdm_distance(bhv_rdm, fmri_rdms[sub, i, j, k], rescale=rescale)

                    print(corrs[sub, i, j, k])

    return corrs


' a function for calculating the Similarity/Correlation Cosfficient between behavioral EEG/MEG/fNIRS and fMRI data (searchlight) '

def eegANDfmri_corr(eeg_data, fmri_data, chl_opt=0, ksize=[3, 3, 3], strides=[1, 1, 1], sub_result=1, method="spearman", fisherz=False, rescale=False, permutation=False, iter=5000):

    """
    Calculate the Similarities between EEG/MEG/fNIRS data and fMRI data for searchligt

    Parameters
    ----------
    eeg_data : array
        The EEG/MEG/fNIRS data.
        The shape of EEGdata must be [n_cons, n_subs, n_trials, n_chls, n_ts].
        n_cons, n_subs, n_trials, n_chls & n_ts represent the number of conidtions, the number of subjects, the number
        of trials, the number of channels & the number of time-points, respectively.
    fmri_data : array
        The fmri data.
        The shape of fmri_data must be [n_cons, n_subs, nx, ny, nz]. n_cons, nx, ny, nz represent the number of
        conditions, the number of subs & the size of fMRI-img, respectively.
    chl_opt : int 0 or 1. Default is 0.
        Calculate the RDM & similarities for each channel or not.
        If chl_opt=0, calculating based on all channels' data.
        If chl_opt=1, calculating based on each channel's data respectively.
    ksize : array or list [kx, ky, kz]. Default is [3, 3, 3].
        The size of the calculation unit for searchlight.
        kx, ky, kz represent the number of voxels along the x, y, z axis.
    strides : array or list [sx, sy, sz]. Default is [1, 1, 1].
        The strides for calculating along the x, y, z axis.
    sub_result: int 0 or 1. Default is 1.
        Return the subject-result or average-result.
        If sub_result=0, return the average result.
        If sub_result=1, return the results of each subject.
    method : string 'spearman' or 'pearson' or 'kendall' or 'similarity' or 'distance'. Default is 'spearman'.
        The method to calculate the similarities.
        If method='spearman', calculate the Spearman Correlations. If method='pearson', calculate the Pearson
        Correlations. If methd='kendall', calculate the Kendall tau Correlations. If method='similarity', calculate the
        Cosine Similarities. If method='distance', calculate the Euclidean Distances.
    fisherz : bool True or False. Default is False.
        Do the Fisher-Z transform of the RDMs or not.
    rescale : bool True or False.
        Rescale the values in RDM or not.
        Here, the maximum-minimum method is used to rescale the values except for the values on the diagonal.
    permutation : bool True or False. Default is False.
        Use permutation test or not.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    corrs : array
        The similarities between EEG/MEG/fNIRS data and fMRI data for searchlight.
        If chl_opt=1 & sub_result=1, the shape of corrs is [n_subs, n_chls, n_x, n_y, n_z, 2]. n_subs, n_x, n_y, n_z
        represent the number of subjects, the number of calculation units for searchlight along the x, y, z axis and 2
        represents a r-value and a p-value.
        If chl_opt=1 & sub_result=0, the shape of corrs is [n_chls, n_x, n_y, n_z, 2]. n_x, n_y, n_z represent the
        number of calculation units for searchlight along the x, y, z axis and 2 represents a r-value and a p-value.
        If chl_opt=0 & sub_result=1, the shape of RDMs is [n_subs, n_x, n_y, n_z, 2]. n_subs, n_x, n_y, n_z represent
        the number of subjects, the number of calculation units for searchlight along the x, y, z axis and 2 represents
        a r-value and a p-value.
        If chl_opt=0 & sub_result=0, the shape of corrs is [n_x, n_y, n_z, 2]. n_x, n_y, n_z represent the number of
        calculation units for searchlight along the x, y, z axis and 2 represents a r-value and a p-value.
    """

    # get the number of subjects
    subs = np.shape(fmri_data)[1]

    # get the size of the fMRI-img
    nx = np.shape(fmri_data)[2]
    ny = np.shape(fmri_data)[3]
    nz = np.shape(fmri_data)[4]

    # the size of the calculation units for searchlight
    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    # strides for calculating along the x, y, z axis
    sx = strides[0]
    sy = strides[1]
    sz = strides[0]

    # calculate the number of the calculation units in the x, y, z directions
    n_x = int((nx - kx) / sx) + 1
    n_y = int((ny - ky) / sy) + 1
    n_z = int((nz - kz) / sz) + 1

    # get the number of channels
    chls = np.shape(eeg_data)[3]

    # calculate the fmri_rdms for searchlight
    fmri_rdms = fmriRDM(fmri_data, ksize=ksize, strides=strides)

    # calculate the eeg_rdms
    eeg_rdms = eegRDM(eeg_data, sub_opt=sub_result, chl_opt=chl_opt)

    # chl_opt=1

    if chl_opt == 1:

        # sub_result=1
        if sub_result == 1:

            # chl_opt=1 & sub_result=1

            # initialize the corrs
            corrs = np.full([subs, chls, n_x, n_y, n_z, 2], np.nan)

            # calculate the corrs
            for sub in range(subs):
                for j in range(n_x):
                    for k in range(n_y):
                        for l in range(n_z):
                            for i in range(chls):

                                if method == "spearman":
                                    corrs[sub, i, j, k, l] = rdm_correlation_spearman(eeg_rdms[sub, i], fmri_rdms[sub, j, k, l], fisherz=fisherz,
                                                                                 rescale=rescale, permutation=permutation, iter=iter)
                                elif method == "pearson":
                                    corrs[sub, i, j, k, l] = rdm_correlation_pearson(eeg_rdms[sub, i], fmri_rdms[sub, j, k, l], fisherz=fisherz,
                                                                                rescale=rescale, permutation=permutation, iter=iter)
                                elif method == "kendall":
                                    corrs[sub, i, j, k, l] = rdm_correlation_kendall(eeg_rdms[sub, i], fmri_rdms[sub, j, k, l], fisherz=fisherz,
                                                                                rescale=rescale, permutation=permutation, iter=iter)
                                elif method == "similarity":
                                    corrs[sub, i, j, k, l, 0] = rdm_similarity(eeg_rdms[sub, i], fmri_rdms[sub, j, k, l],
                                                                          rescale=rescale)
                                elif method == "distance":
                                    corrs[sub, i, j, k, l, 0] = rdm_distance(eeg_rdms[sub, i], fmri_rdms[sub, i, j, k],
                                                                        rescale=rescale)

            return np.abs(corrs)

        # sub_result=0

        # chl_opt=1 & sub_result=0

        # initialize the corrs
        corrs = np.full([chls, n_x, n_y, n_z, 2], np.nan)

        # calculate the corrs
        for j in range(n_x):
            for k in range(n_y):
                for l in range(n_z):
                    for i in range(chls):

                        if method == "spearman":
                            corrs[i, j, k, l] = rdm_correlation_spearman(eeg_rdms[i], fmri_rdms[j, k, l], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                        elif method == "pearson":
                            corrs[i, j, k, l] = rdm_correlation_pearson(eeg_rdms[i], fmri_rdms[j, k, l], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                        elif method == "kendall":
                            corrs[i, j, k, l] = rdm_correlation_kendall(eeg_rdms[i], fmri_rdms[j, k, l], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                        elif method == "similarity":
                            corrs[i, j, k, l, 0] = rdm_similarity(eeg_rdms[i], fmri_rdms[j, k, l], rescale=rescale)
                        elif method == "distance":
                            corrs[i, j, k, l, 0] = rdm_distance(eeg_rdms[i], fmri_rdms[i, j, k], rescale=rescale)

        return np.abs(corrs)

    # chl_opt=0

    # sub_result=1

    if sub_result == 1:

        # chl_opt=0 & sub_result=1

        # initialize the corrs
        corrs = np.full([subs, n_x, n_y, n_z, 2], np.nan)

        # calculate the corrs
        for i in range(subs):
            for j in range(n_x):
                for k in range(n_y):
                    for l in range(n_z):

                        if method == "spearman":
                            corrs[i, j, k, l] = rdm_correlation_spearman(eeg_rdms[i], fmri_rdms[j, k, l], fisherz=fisherz,
                                                                         rescale=rescale, permutation=permutation, iter=iter)
                        elif method == "pearson":
                            corrs[i, j, k, l] = rdm_correlation_pearson(eeg_rdms[i], fmri_rdms[j, k, l], fisherz=fisherz,
                                                                        rescale=rescale, permutation=permutation, iter=iter)
                        elif method == "kendall":
                            corrs[i, j, k, l] = rdm_correlation_kendall(eeg_rdms[i], fmri_rdms[j, k, l], fisherz=fisherz,
                                                                        rescale=rescale, permutation=permutation, iter=iter)
                        elif method == "similarity":
                            corrs[i, j, k, l, 0] = rdm_similarity(eeg_rdms[i], fmri_rdms[j, k, l], rescale=rescale)
                        elif method == "distance":
                            corrs[i, j, k, l, 0] = rdm_distance(eeg_rdms[i], fmri_rdms[i, j, k], rescale=rescale)

    # sub_result=0

    # chl_opt=0 & sub_result=0

    # initialize the corrs
    corrs = np.full([n_x, n_y, n_z, 2], np.nan)

    # calculate the corrs
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):

                if method == "spearman":
                    corrs[i, j, k] = rdm_correlation_spearman(eeg_rdms, fmri_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "pearson":
                    corrs[i, j, k] = rdm_correlation_pearson(eeg_rdms, fmri_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "kendall":
                    corrs[i, j, k] = rdm_correlation_kendall(eeg_rdms, fmri_rdms[i, j, k], fisherz=fisherz, rescale=rescale, permutation=permutation, iter=iter)
                elif method == "similarity":
                    corrs[i, j, k, 0] = rdm_similarity(eeg_rdms, fmri_rdms[i, j, k], rescale=rescale)
                elif method == "distance":
                    corrs[i, j, k, 0] = rdm_distance(eeg_rdms, fmri_rdms[i, j, k], rescale=rescale)

    return np.abs(corrs)