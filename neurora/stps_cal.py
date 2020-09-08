# -*- coding: utf-8 -*-

' a module for calculating the spatiotemporal pattern similarity based on neural data '

__author__ = 'Zitong Lu'

import numpy as np
from scipy.stats import pearsonr
import math

np.seterr(divide='ignore', invalid='ignore')


' a function for calculating the spatiotemporal pattern similarities (STPS) '

def stps(data, label_item, label_rf, time_win=20, time_step=1):

    """
    Calculate the spatiotemporal pattern similarities (STPS)

    Parameters
    ----------
    data : array
        The neural data.
        The shape of data must be [n_subs, n_trials, n_chls, n_ts]. n_subs, n_trials, n_chls and n_ts represent the
        number of subjects, the number of trials, the number of channels or regions and the number of time-points.
    label_item : array or list.
        The label of trials.
        The shape of label_wibi must be [n_trials]. n_trials represents the number of trials.
    label_rf : array or list.
        The label of trials: Remembered (0) or Forgot (1).
        The shape of label_rf must be [n_trials]. n_trials represents the number of trials. If the trial i is a
        remembered trial, label_rf[i]=0. If the trial j is a forgot trial, label_rf[j]=0.
    time_win : int. Default is 20.
        Set a time-window for calculating the STPS for different time-points.
        If time_win=20, that means each calculation process based on 20 time-points.
    time_step : int. Default is 1.
        The time step size for each time of calculating.

    Returns
    -------
    stps : array.
        The STPS.
        The shape of stps is [n_subs, 8, n_chls, int((n_ts-time_win)/time_step)+1]. 8 represents eight different
        conditions: 0: Within-Item, 1: Between-Item, 2: Remembered, 3: Forgot, 4: Within-Item&Remembered,
        5: Within-Item&Forgot, 6: Between-Item&Remembered, 7: Between-Item&Forgot.
    """

    # get the number of subjects, trials, channels/regions & time-points
    subs, trials, chls, ts = np.shape(data)

    # the time-points for calculating STPS
    ts = int((ts - time_win) / time_step) + 1

    # initialize the STPS
    stps = np.zeros([subs, 8, chls, ts], dtype=np.float)

    for sub in range(subs):

        # initialize the STPS for each subject
        sub_stps = np.zeros([8, chls, ts], dtype=np.float)

        for i in range(chls):
            for j in range(ts):

                trials_data = data[sub, :, i, ts*time_step:ts*time_step+time_win]

                corr_mat = np.zeros([trials, trials], dtype=np.float)

                index = np.zeros([8], dtype=np.int)

                for k in range(trials):
                    for l in range(trials):

                        corr_mat[k, l] = pearsonr(trials_data[k], trials_data[l])[0]

                        if k < l:

                            if label_item[k] == label_item[l]:
                                index[0] = index[0] + 1

                                if label_rf[k] == 0 and label_rf[l] == 0:
                                    index[4] = index[4] + 1
                                if label_rf[k] == 1 and label_rf[l] == 1:
                                    index[5] = index[5] + 1

                            if label_item[k] != label_item[l]:
                                index[1] = index[1] + 1

                                if label_rf[k] == 0 and label_rf[l] == 0:
                                    index[6] = index[6] + 1
                                if label_rf[k] == 1 and label_rf[l] == 1:
                                    index[7] = index[7] + 1

                            if label_rf[k] == 0 and label_rf[l] == 0:
                                index[2] = index[2] + 1

                            if label_rf[k] == 1 and label_rf[l] == 1:
                                index[3] = index[3] + 1

                r0 = np.zeros([index[0]], dtype=np.float)
                r1 = np.zeros([index[1]], dtype=np.float)
                r2 = np.zeros([index[2]], dtype=np.float)
                r3 = np.zeros([index[3]], dtype=np.float)
                r4 = np.zeros([index[4]], dtype=np.float)
                r5 = np.zeros([index[5]], dtype=np.float)
                r6 = np.zeros([index[6]], dtype=np.float)
                r7 = np.zeros([index[7]], dtype=np.float)

                index = np.zeros([8], dtype=np.int)

                for k in range(trials):
                    for l in range(trials):

                        if k < l:

                            if label_item[k] == label_item[l]:
                                r0[index[0]] = corr_mat[k, l]
                                index[0] = index[0] + 1

                                if label_rf[k] == 0 and label_rf[l] == 0:
                                    r4[index[4]] = corr_mat[k, l]
                                    index[4] = index[4] + 1
                                if label_rf[k] == 1 and label_rf[l] == 1:
                                    r5[index[5]] = corr_mat[k, l]
                                    index[5] = index[5] + 1

                            if label_item[k] != label_item[l]:
                                r1[index[1]] = corr_mat[k, l]
                                index[1] = index[1] + 1

                                if label_rf[k] == 0 and label_rf[l] == 0:
                                    r6[index[6]] = corr_mat[k, l]
                                    index[6] = index[6] + 1
                                if label_rf[k] == 1 and label_rf[l] == 1:
                                    r7[index[7]] = corr_mat[k, l]
                                    index[7] = index[7] + 1

                            if label_rf[k] == 0 and label_rf[l] == 0:
                                r2[index[2]] = corr_mat[k, l]
                                index[2] = index[2] + 1

                            if label_rf[k] == 1 and label_rf[l] == 1:
                                r3[index[3]] = corr_mat[k, l]
                                index[3] = index[3] + 1

                sub_stps[0, i, j] = np.average(r0)
                sub_stps[1, i, j] = np.average(r1)
                sub_stps[2, i, j] = np.average(r2)
                sub_stps[3, i, j] = np.average(r3)
                sub_stps[4, i, j] = np.average(r4)
                sub_stps[5, i, j] = np.average(r5)
                sub_stps[6, i, j] = np.average(r6)
                sub_stps[7, i, j] = np.average(r7)

        stps[sub] = sub_stps

    return stps


' a function for calculating the spatiotemporal pattern similarities (STPS) for fMRI (searchlight) '

def stps_fmri(fmri_data, label_item, label_rf, ksize=[3, 3, 3], strides=[1, 1, 1]):

    """
    Calculate the spatiotemporal pattern similarities (STPS) for fMRI (searchlight)

    Parameters
    ----------
    fmri_data : array
        The fMRI data.
        The shape of fmri_data must be [n_subs, n_trials, nx, ny, nz]. n_subs, n_trials, nx, ny, nz represent the number
        of subjects, the number of trials & the size of fMRI-img, respectively.
    label_item : array or list.
        The label of trials.
        The shape of label_wibi must be [n_trials]. n_trials represents the number of trials.
    label_rf : array or list.
        The label of trials: Remembered (0) or Forgot (1).
        The shape of label_rf must be [n_trials]. n_trials represents the number of trials. If the trial i is a
        remembered trial, label_rf[i]=0. If the trial j is a forgot trial, label_rf[j]=0.
    ksize : array or list [kx, ky, kz]. Default is [3, 3, 3].
        The size of the calculation unit for searchlight.
        kx, ky, kz represent the number of voxels along the x, y, z axis.
    strides : array or list [sx, sy, sz]. Default is [1, 1, 1].
        The strides for calculating along the x, y, z axis.

    Returns
    -------
    stps : array.
        The STPS.
        The shape of stps is [n_subs, 8, n_x, n_y, n_z]. 8 represents eight different
        conditions: 0: Within-Item, 1: Between-Item, 2: Remembered, 3: Forgot, 4: Within-Item&Remembered,
        5: Within-Item&Forgot, 6: Between-Item&Remembered, 7: Between-Item&Forgot. n_x, n_y, n_z represent the number
        of calculation units for searchlight along the x, y, z axis.

    Notes
    -----
    The size of the calculation units should at least be [3, 3, 3].
    """


    # get the number of subjects, trials and the size of the fMRI-img
    subs, trials, nx, ny, nz = np.shape(fmri_data)

    # the size of the calculation units for searchlight
    kx = ksize[0]
    ky = ksize[1]
    kz = ksize[2]

    if kx + ky + kz < 9:
        return print("The size of the calculation units is too small.")

    # strides for calculating along the x, y, z axis
    sx = strides[0]
    sy = strides[1]
    sz = strides[2]

    # calculate the number of the calculation units
    n_x = int((nx - kx) / sx) + 1
    n_y = int((ny - ky) / sy) + 1
    n_z = int((nz - kz) / sz) + 1

    # initialize the STPS
    stps = np.zeros([subs, 8, n_x, n_y, n_z], dtype=np.float)

    for sub in range(subs):

        # initialize the STPS for each subject
        sub_stps = np.zeros([8, n_x, n_y, n_z], dtype=np.float)

        for x in range(n_x):
            for y in range(n_y):
                for z in range(n_z):

                    trials_data = fmri_data[sub, :, x*sx:x*sx+kx, y*sy:y*sy+ky, z*sz:z*sz+kz]
                    trials_data = np.reshape(trials_data, [trials, kx*ky*kz])

                    corr_mat = np.zeros([trials, trials], dtype=np.float)

                    index = np.zeros([8], dtype=np.int)

                    for k in range(trials):
                        for l in range(trials):

                            corr_mat[k, l] = pearsonr(trials_data[k], trials_data[l])[0]

                            if k < l:

                                if label_item[k] == label_item[l]:
                                    index[0] = index[0] + 1

                                    if label_rf[k] == 0 and label_rf[l] == 0:
                                        index[4] = index[4] + 1
                                    if label_rf[k] == 1 and label_rf[l] == 1:
                                        index[5] = index[5] + 1

                                if label_item[k] != label_item[l]:
                                    index[1] = index[1] + 1

                                    if label_rf[k] == 0 and label_rf[l] == 0:
                                        index[6] = index[6] + 1
                                    if label_rf[k] == 1 and label_rf[l] == 1:
                                        index[7] = index[7] + 1

                                if label_rf[k] == 0 and label_rf[l] == 0:
                                    index[2] = index[2] + 1

                                if label_rf[k] == 1 and label_rf[l] == 1:
                                    index[3] = index[3] + 1

                    r0 = np.zeros([index[0]], dtype=np.float)
                    r1 = np.zeros([index[1]], dtype=np.float)
                    r2 = np.zeros([index[2]], dtype=np.float)
                    r3 = np.zeros([index[3]], dtype=np.float)
                    r4 = np.zeros([index[4]], dtype=np.float)
                    r5 = np.zeros([index[5]], dtype=np.float)
                    r6 = np.zeros([index[6]], dtype=np.float)
                    r7 = np.zeros([index[7]], dtype=np.float)

                    index = np.zeros([8], dtype=np.int)

                    for k in range(trials):
                        for l in range(trials):

                            if k < l:

                                if label_item[k] == label_item[l]:
                                    r0[index[0]] = corr_mat[k, l]
                                    index[0] = index[0] + 1

                                    if label_rf[k] == 0 and label_rf[l] == 0:
                                        r4[index[4]] = corr_mat[k, l]
                                        index[4] = index[4] + 1
                                    if label_rf[k] == 1 and label_rf[l] == 1:
                                        r5[index[5]] = corr_mat[k, l]
                                        index[5] = index[5] + 1

                                if label_item[k] != label_item[l]:
                                    r1[index[1]] = corr_mat[k, l]
                                    index[1] = index[1] + 1

                                    if label_rf[k] == 0 and label_rf[l] == 0:
                                        r6[index[6]] = corr_mat[k, l]
                                        index[6] = index[6] + 1
                                    if label_rf[k] == 1 and label_rf[l] == 1:
                                        r7[index[7]] = corr_mat[k, l]
                                        index[7] = index[7] + 1

                                if label_rf[k] == 0 and label_rf[l] == 0:
                                    r2[index[2]] = corr_mat[k, l]
                                    index[2] = index[2] + 1

                                if label_rf[k] == 1 and label_rf[l] == 1:
                                    r3[index[3]] = corr_mat[k, l]
                                    index[3] = index[3] + 1

                    sub_stps[0, x, y, z] = np.average(r0)
                    sub_stps[1, x, y, z] = np.average(r1)
                    sub_stps[2, x, y, z] = np.average(r2)
                    sub_stps[3, x, y, z] = np.average(r3)
                    sub_stps[4, x, y, z] = np.average(r4)
                    sub_stps[5, x, y, z] = np.average(r5)
                    sub_stps[6, x, y, z] = np.average(r6)
                    sub_stps[7, x, y, z] = np.average(r7)

        stps[sub] = sub_stps

    return stps


' a function for calculating the spatiotemporal pattern similarities (STPS) for fMRI (for ROI) '

def stps_fmri_roi(fmri_data, mask_data, label_item, label_rf):
    """
    Calculate the spatiotemporal pattern similarities (STPS) for fMRI (for ROI)

    Parameters
    ----------
    fmri_data : array
        The fmri data.
        The shape of fmri_data must be [n_subs, n_trials, nx, ny, nz]. n_subs, n_trials, nx, ny, nz represent the number
        of subjects, the number of trials & the size of fMRI-img, respectively.
    mask_data : array [nx, ny, nz].
        The mask data for region of interest (ROI).
        The size of the fMRI-img. nx, ny, nz represent the number of voxels along the x, y, z axis.
    label_item : array or list.
        The label of trials.
        The shape of label_wibi must be [n_trials]. n_trials represents the number of trials.
    label_rf : array or list.
        The label of trials: Remembered (0) or Forgot (1).
        The shape of label_rf must be [n_trials]. n_trials represents the number of trials. If the trial i is a
        remembered trial, label_rf[i]=0. If the trial j is a forgot trial, label_rf[j]=0.

    Returns
    -------
    stps : array.
        The STPS.
        The shape of stps is [n_subs, 8]. 8 represents eight different conditions: 0: Within-Item, 1: Between-Item,
        2: Remembered, 3: Forgot, 4: Within-Item&Remembered, 5: Within-Item&Forgot, 6: Between-Item&Remembered,
        7: Between-Item&Forgot.

    Notes
    -----
    The size of the calculation units should at least be [3, 3, 3].
    """

    # get the number of subjects, trials and the size of the fMRI-img
    subs, trials, nx, ny, nz = np.shape(fmri_data)

    # record the number of valid voxels in ROI
    nmask = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                # not 0 or NaN
                if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False):
                    nmask = nmask + 1

    # initialize the data for calculating the ISC
    data = np.full([subs, trials, nmask], np.nan)

    # assignment
    for sub in range(subs):
        for i in range(trials):

            # record the index of the valid voxels for calculating
            n = 0
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):

                        # not 0 or NaN
                        if (mask_data[i, j, k] != 0) and (math.isnan(mask_data[i, j, k]) == False):
                            data[sub, i, n] = fmri_data[sub, i, i, j, k]
                            n = n + 1

    # initialize the STPS
    stps = np.zeros([subs, 8], dtype=np.float)

    for sub in range(subs):

        sub_stps = np.zeros([8], dtype=np.float)

        trials_data = data[sub]

        corr_mat = np.zeros([trials, trials], dtype=np.float)

        index = np.zeros([8], dtype=np.int)

        for k in range(trials):
            for l in range(trials):

                corr_mat[k, l] = pearsonr(trials_data[k], trials_data[l])[0]

                if k < l:

                    if label_item[k] == label_item[l]:
                        index[0] = index[0] + 1

                        if label_rf[k] == 0 and label_rf[l] == 0:
                            index[4] = index[4] + 1
                        if label_rf[k] == 1 and label_rf[l] == 1:
                            index[5] = index[5] + 1

                    if label_item[k] != label_item[l]:
                        index[1] = index[1] + 1

                        if label_rf[k] == 0 and label_rf[l] == 0:
                            index[6] = index[6] + 1
                        if label_rf[k] == 1 and label_rf[l] == 1:
                            index[7] = index[7] + 1

                    if label_rf[k] == 0 and label_rf[l] == 0:
                        index[2] = index[2] + 1

                    if label_rf[k] == 1 and label_rf[l] == 1:
                        index[3] = index[3] + 1

        r0 = np.zeros([index[0]], dtype=np.float)
        r1 = np.zeros([index[1]], dtype=np.float)
        r2 = np.zeros([index[2]], dtype=np.float)
        r3 = np.zeros([index[3]], dtype=np.float)
        r4 = np.zeros([index[4]], dtype=np.float)
        r5 = np.zeros([index[5]], dtype=np.float)
        r6 = np.zeros([index[6]], dtype=np.float)
        r7 = np.zeros([index[7]], dtype=np.float)

        index = np.zeros([8], dtype=np.int)

        for k in range(trials):
            for l in range(trials):

                if k < l:

                    if label_item[k] == label_item[l]:
                        r0[index[0]] = corr_mat[k, l]
                        index[0] = index[0] + 1

                        if label_rf[k] == 0 and label_rf[l] == 0:
                            r4[index[4]] = corr_mat[k, l]
                            index[4] = index[4] + 1
                        if label_rf[k] == 1 and label_rf[l] == 1:
                            r5[index[5]] = corr_mat[k, l]
                            index[5] = index[5] + 1

                    if label_item[k] != label_item[l]:
                        r1[index[1]] = corr_mat[k, l]
                        index[1] = index[1] + 1

                        if label_rf[k] == 0 and label_rf[l] == 0:
                            r6[index[6]] = corr_mat[k, l]
                            index[6] = index[6] + 1
                        if label_rf[k] == 1 and label_rf[l] == 1:
                            r7[index[7]] = corr_mat[k, l]
                            index[7] = index[7] + 1

                    if label_rf[k] == 0 and label_rf[l] == 0:
                        r2[index[2]] = corr_mat[k, l]
                        index[2] = index[2] + 1

                    if label_rf[k] == 1 and label_rf[l] == 1:
                        r3[index[3]] = corr_mat[k, l]
                        index[3] = index[3] + 1

        sub_stps[0] = np.average(r0)
        sub_stps[1] = np.average(r1)
        sub_stps[2] = np.average(r2)
        sub_stps[3] = np.average(r3)
        sub_stps[4] = np.average(r4)
        sub_stps[5] = np.average(r5)
        sub_stps[6] = np.average(r6)
        sub_stps[7] = np.average(r7)

        stps[sub] = sub_stps

    return stps