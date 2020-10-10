# -*- coding: utf-8 -*-

' a module for calculating the Inter Subject Correlation based on neural data '

__author__ = 'Zitong Lu'

import numpy as np
import math
from scipy.stats import pearsonr

np.seterr(divide='ignore', invalid='ignore')


' a function for calculating the inter subject correlation (ISC) '

def isc(data, time_win=5, time_step=5):

    """
    Calculate the inter subject correlation (ISC)

    Parameters
    ----------
    data : array
        The neural data.
        The shape of data must be [n_subs, n_chls, n_ts]. n_subs, n_chls, n_ts represent the number of subjects, the
        number of channels and the number of time-points.
    time_win : int. Default is 5.
        Set a time-window for calculating the STPS for different time-points.
        If time_win=5, that means each calculation process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.

    Returns
    -------
    isc : array
        The ISC.
        The shape of isc is [n_subs!/(2!*(n_subs-2)!), n_chls, int((n_ts-time_win)/time_step)+1, 2]. n_subs, n_chls,
        n_ts represent the number of subjects, the number of channels and the number of time-points. 2 represents a
        r-value and a p-value.
    """

    # get the number of subjects, channels, time-points
    subs, chls, ts = np.shape(data)

    # the time-points for calculating the ISC
    ts = int((ts - time_win) / time_step) + 1

    # the number of pairs among n_subs
    if subs > 2:
        n = int(math.factorial(subs)/(2*math.factorial(subs-2)))
    if subs == 2:
        n = 1

    # initialize the corrs
    isc = np.zeros([n, chls, ts, 2], dtype=np.float)

    nindex = 0
    # calculate the ISC
    for i in range(subs):
        for j in range(subs):

            if i < j:

                for k in range(chls):
                    for l in range(ts):

                        rp = pearsonr(data[i, k, l*time_step:l*time_step+time_win], data[j, k, l*time_step:l*time_step+time_win])
                        isc[nindex, k, l] = rp

                nindex = nindex + 1

    return isc


' a function for calculating the inter subject correlation (ISC) for fMRI (searchlight) '

def isc_fmri(fmri_data, ksize=[3, 3, 3], strides=[1, 1, 1]):

    """
    Calculate the inter subject correlation (ISC) for fMRI (searchlight)

    Parameters
    ----------
    fmri_data : array
        The fmri data.
        The shape of fmri_data must be [n_ts, n_subs, nx, ny, nz]. n_ts, nx, ny, nz represent the number of time-points,
        the number of subs & the size of fMRI-img, respectively.
    ksize : array or list [kx, ky, kz]. Default is [3, 3, 3].
        The size of the calculation unit for searchlight.
        kx, ky, kz represent the number of voxels along the x, y, z axis.
    strides : array or list [sx, sy, sz]. Default is [1, 1, 1].
        The strides for calculating along the x, y, z axis.

    Returns
    -------
    isc : array
        The ISC.
        The shape of isc is [n_ts, n_subs!/(2!*(n_subs-2)!), n_x, n_y, n_z, 2]. n_ts, n_subs, n_x, n_y, n_z represent
        the number of time-points, the number of subjects, the number of calculation units for searchlight along the x,
        y, z axis. 2 represent a r-value and a p-value.

    Notes
    -----
    The size of the calculation units should at least be [3, 3, 3].
    """

    # get the number of time-points, subjects and the size of the fMRI-img
    nts, nsubs, nx, ny, nz = np.shape(fmri_data)

    # the size of the calculation units for searchlight
    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    if kx+ky+kz < 9:
        return print("The size of the calculation units is too small.")

    # strides for calculating along the x, y, z axis
    sx = strides[0]
    sy = strides[1]
    sz = strides[2]

    # calculate the number of the calculation units
    n_x = int((nx - kx) / sx) + 1
    n_y = int((ny - ky) / sy) + 1
    n_z = int((nz - kz) / sz) + 1

    # initialize the data for calculating the ISC
    data = np.full([nts, nsubs, n_x, n_y, n_z, kx * ky * kz], np.nan)

    # assignment
    for t in range(nts):
        for sub in range(nsubs):
            for x in range(n_x):
                for y in range(n_y):
                    for z in range(n_z):

                        # record the index in a calculation unit
                        index = 0
                        for k1 in range(kx):
                            for k2 in range(ky):
                                for k3 in range(kz):
                                    data[t, sub, x, y, z, index] = fmri_data[t, sub, x + k1, y + k2, z + k3]

                                    index = index + 1

    # the number of pairs among n_subs
    if nsubs > 2:
        n = int(math.factorial(nsubs) / (2 * math.factorial(nsubs - 2)))
    if nsubs == 2:
        n = 1

    # initialize the ISC
    subisc = np.full([nts, n, n_x, n_y, n_z, 2], np.nan)

    # calculate the ISC
    for t in range(nts):

        nindex = 0
        for i in range(nsubs):
            for j in range(nsubs):

                if i < j:

                    for x in range(n_x):
                        for y in range(n_y):
                            for z in range(n_z):

                                # no NaN
                                if (np.isnan(data[t, i, x, y, z]).any() == False) and (np.isnan(data[t, j, x, y, z]).any() == False):
                                    # calculate the Pearson Coefficient and absolute the result
                                    subisc[t, nindex, x, y, z] = pearsonr(data[t, i, x, y, z], data[t, j, x, y, z])

                    nindex = nindex + 1

    return subisc


' a function for calculating the inter subject correlation (ISC) for fMRI (for ROI) '

def isc_fmri_roi(fmri_data, mask_data):

    """
    Calculate the inter subject correlation (ISC) for fMRI (for ROI)

    Parameters
    ----------
    fmri_data : array
        The fmri data.
        The shape of fmri_data must be [n_ts, n_subs, nx, ny, nz]. n_ts, nx, ny, nz represent the number of time-points,
        the number of subs & the size of fMRI-img, respectively.
    mask_data : array [nx, ny, nz].
        The mask data for region of interest (ROI).
        The size of the fMRI-img. nx, ny, nz represent the number of voxels along the x, y, z axis.

    Returns
    -------
    isc : array
        The ISC.
        The shape of corrs is [n_ts, n_subs!/(2!*(n_subs-2)!), 2]. n_ts, n_subs represent  the number of time-points,
        the number of subjects. 2 represent a r-value and a p-value.

    Notes
    -----
    The size of the calculation units should at least be [3, 3, 3].
    """

    # get the number of time-points, subjects and the size of the fMRI-img
    nts, nsubs, nx, ny, nz = np.shape(fmri_data)

    # record the number of valid voxels in ROI
    nmask = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                # not 0 or NaN
                if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False):
                    nmask = nmask + 1

    # initialize the data for calculating the ISC
    data = np.full([nts, nsubs, nmask], np.nan)

    # assignment
    for t in range(nts):
        for sub in range(nsubs):

            # record the index of the valid voxels for calculating
            n = 0
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):

                        # not 0 or NaN
                        if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False):
                            data[t, sub, n] = fmri_data[t, sub, i, j, k]
                            n = n + 1

    # the number of pairs among n_subs
    if nsubs > 2:
        n = int(math.factorial(nsubs) / (2 * math.factorial(nsubs - 2)))
    if nsubs == 2:
        n = 1

    # initialize the ISC
    subisc = np.full([nts, n, 2], np.nan)

    # calculate the ISC
    for t in range(nts):

        nindex = 0
        for i in range(nsubs):
            for j in range(nsubs):

                if i < j:

                    if (np.isnan(data[t, i]).any() == False) and (np.isnan(data[t, j]).any() == False):
                                    # calculate the Pearson Coefficient and absolute the result
                                    subisc[t, nindex] = pearsonr(data[t, i], data[t, j])

                    nindex = nindex + 1

    return subisc



