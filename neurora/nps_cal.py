# -*- coding: utf-8 -*-

' a module for calculating the neural pattern similarity based neural data '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import pearsonr
import math

np.seterr(divide='ignore', invalid='ignore')


' a function for calculating the neural pattern similarity for EEG-like data '

def nps(data, time_win=5, sub_opt=0):

    """
    Calculate the Neural Representational Similarity (NPS) for EEG-like data

    Parameters
    ----------
    data : array
        The EEG-like neural data.
        The shape of data must be [2, n_subs, n_trials, n_chls, n_ts].
        2 presents 2 different conditions. n_subs, n_trials, n_chls & n_ts represent the number of subjects,
        the number of trials, the number of channels & the number of time-points, respectively.
    time_win : int. Default is 5.
        Set a time-window for calculating the NPS for different time-points.
        If time_win=5, that means each calculation process based on 5 time-points.
        This is also a processing of downsampling.
    sub_opt : int 0 or 1. Default is 0.
        Calculate the NPS for each subject or not.
        If sub_opt=0, calculate the NPS based on all data.
        If sub_opt=1, calculate the NPS based on each subject's data

    Returns
    -------
    NPS : array
        The EEG-like NPS.
        If sub_opt=0, the shape of NPS is [n_chls, int(n_ts/time_win), 2].
        If sub_opt=1, the shape of NPS is [n_subs, n_chls, int(n_ts/time_win), 2].
        2 representation a r-value and a p-value.
    """

    # get the number of subjects, trials, channels & time-points
    nsubs, ntrials, nchls, nts = data.shape[1:]

    # average the trials
    avgdata = np.average(data, axis=2)

    # the time-points for calculating NPS
    ts = int(nts/time_win)

    # sub_opt=1
    if sub_opt == 1:

        # initialize the NPS
        nps = np.zeros([nsubs, nchls, ts, 2])

        # calculate the NPS
        for sub in range(nsubs):
            for i in range(nchls):
                for j in range(ts):

                    # average the time_win
                    data1 = avgdata[0, sub, i, j*time_win:j*time_win+time_win]
                    data2 = avgdata[1, sub, i, j*time_win:j*time_win+time_win]
                    # calculate the Pearson Coefficient
                    nps[sub, i, j] = pearsonr(data1, data2)

        return nps

    # if sub_opt == 0

    # initialize the NPS
    nps = np.zeros([nchls, ts, 2])

    # calculate the NPS
    for i in range(nchls):
        for j in range(ts):

            # average the time_win
            data1 = avgdata[0, :, i, j*time_win:j*time_win+time_win]
            data2 = avgdata[1, :, i, j*time_win:j*time_win+time_win]

            # flatten the data
            data1 = np.reshape(data1, nsubs*time_win)
            data2 = np.reshape(data2, nsubs*time_win)

            # calculate the Pearson Coefficient
            nps[i, j] = pearsonr(data1, data2)

    return nps


' a function for calculating the neural pattern similarity for fMRI data ( for searchlight) '

def nps_fmri(fmri_data, ksize=[3, 3, 3], strides=[1, 1, 1]):

    """
    Calculate the Neural Representational Similarity (NPS) for fMRI data for searchlight

    Parameters
    ----------
    fmri_data : array [nx, ny, nz].
        The fmri data.
        The shape of fmri_data must be [n_cons, n_chls, nx, ny, nz].
        n_cons, n_chls, nx, ny, nz represent the number of conidtions, the number of channels &
        the size of fMRI-img, respectively.
    ksize : array or list [kx, ky, kz]. Default is [3, 3, 3].
        The size of the fMRI-img. nx, ny, nz represent the number of voxels along the x, y, z axis
    strides : array or list [sx, sy, sz]. Default is [1, 1, 1].
        The strides for calculating along the x, y, z axis.

    Returns
    -------
    NPS : array
        The fMRI NPS for searchlight.
        The shape of NPS is [n_x, n_y, n_z, 2]. n_x, n_y, n_z represent the number of calculation units
        for searchlight along the x, y, z axis
    """

    # get the number of subjects and the size of the fMRI-img
    nsubs, nx, ny, nz = np.shape(fmri_data)[1:]

    # the size of the calculation units for searchlight
    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    # strides for calculating along the x, y, z axis
    sx = strides[0]
    sy = strides[1]
    sz = strides[2]

    # calculate the number of the calculation units
    n_x = int((nx - kx) / sx) + 1
    n_y = int((ny - ky) / sy) + 1
    n_z = int((nz - kz) / sz) + 1

    # initialize the data for calculating the NPS
    data = np.full([n_x, n_y, n_z, 2, kx*ky*kz, nsubs], np.nan)

    # assignment
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                for i in range(2):

                    # record the index in a calculation unit
                    index = 0

                    for k1 in range(kx):
                        for k2 in range(ky):
                            for k3 in range(kz):
                                for j in range(nsubs):
                                    data[x, y, z, i, index, j] = fmri_data[i, j, x+k1, y+k2, z+k3]

                                index = index + 1

    # flatten the data under each condition for each calculation unit
    data = np.reshape(data, [n_x, n_y, n_z, 2, kx*ky*kz*nsubs])

    # initialize the NPS
    nps = np.full([n_x, n_y, n_z, 2], np.nan)

    # calculate the NPS
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):

                # no NaN
                if (np.isnan(data[x, y, z, 0]).any() == False) and (np.isnan(data[x, y, z, 1]).any() == False):
                    # calculate the Pearson Coefficient and absolute the result
                    nps[x, y, z] = np.abs(pearsonr(data[x, y, z, 0], data[x, y, z, 1]))

    return nps


' a function for calculating the neural pattern similarity for fMRI data (for ROI) '

def nps_fmri_roi(fmri_data, mask_data):

    """
    Calculate the Neural Representational Similarity (NPS) for fMRI data for ROI

    Parameters
    ----------
    fmri_data : array [nx, ny, nz].
        The fmri data.
        The shape of fmri_data must be [n_cons, n_chls, nx, ny, nz].
        n_cons, n_chls, nx, ny, nz represent the number of conidtions, the number of channels &
        the size of fMRI-img, respectively.
    mask_data : array [nx, ny, nz].
        The mask data for region of interest (ROI)
        The size of the fMRI-img. nx, ny, nz represent the number of voxels along the x, y, z axis

    Returns
    -------
    NPS : array
        The fMRI NPS for ROI.
        The shape of NPS is [2]. 2 representation a r-value and a p-value.
    """

    # get the number of subjects and the size of the fMRI-img
    nsubs, nx, ny, nz = fmri_data.shape[1:]

    # record the number of valid voxels in ROI
    n = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                # not 0 or NaN
                if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False):
                    n = n + 1

    # initialize the data for calculating the NPS
    data = np.zeros([2, nsubs, n], dtype=np.float)

    # assignment
    for p in range(2):
        for q in range(nsubs):

            # record the index of the valid voxels for calculating
            n = 0
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):

                        # not 0 or NaN
                        if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False):
                            data[p, q, n] = fmri_data[p, q, i, j, k]
                            n = n + 1

    # flatten the data
    data = np.reshape(data, [2, nsubs*n])

    # calculate the Pearson Coefficient
    nps = np.abs(pearsonr(data[0], data[1]))

    return nps